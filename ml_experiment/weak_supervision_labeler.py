"""
Weak Supervision 레이블러 (클러스터 분석 기반 개선 버전)

목적:
- 사용자 피드백 없이 자동으로 레이블 생성
- 규칙 기반으로 "이 번역이 신뢰할 수 있는가?" 판단
- 생성된 레이블로 ML 모델 학습

Weak Supervision이란?
- 완벽한 정답(Gold Label)이 없을 때 사용하는 방법
- 여러 규칙(Labeling Function)을 조합해서 레이블 추정
- 규칙이 틀릴 수도 있지만, 여러 규칙을 합치면 꽤 정확함

레이블 정의:
- 1 (GOOD): 신뢰할 수 있는 번역 (용어사전 사용 권장)
- 0 (BAD): 신뢰하기 어려운 번역 (LLM 필요하거나 검토 필요)
- -1 (ABSTAIN): 판단 불가 (규칙이 적용되지 않음)
"""

from math import e
import pandas as pd
import numpy as np
from typing import Callable

from core.config import TranslationFeatures as TF
from core.config import TranslationReason as TR

# 레이블 상수 정의
GOOD = 1      # 신뢰할 수 있음
BAD = 0       # 신뢰하기 어려움
ABSTAIN = -1  # 판단 불가


# ============================================================
# Row 가져오는 Helper 함수
# ============================================================
def _get_row(row, key, default=None):
    v = row.get(key, default)
    
    return default if v is None else v

# ============================================================
# 클러스터 분석 기반 추가 규칙들
# ============================================================
def lf_strong_good_exact_bm25_no_llm(row) -> int:
    """
    규칙 1: 강력한 GOOD 레이블 (정확한 매칭 + BM25 + LLM 없음)
    
    조건: exact match + BM25 match + rank 1 + LLM fallback 없음
    이유: 클러스터 분석에서 가장 이상적인 번역 패턴으로 확인됨
          Cluster 1은 exact 96% + BM25 96%의 조합
    """

    # Exact + BM25 + rank 1등 + LLM 안 씀
    if _get_row(row, TF.IS_EXACT_MATCH, 0) == 1 and _get_row(row, TF.BM25_TOP_RANK_IN_HYBRID, 0) == 1 and _get_row(row, TF.IS_BM25_MATCH, 0) == 1:
        if _get_row(row, TF.IS_LLM_FALLBACK, 0) == 0:
            return GOOD
    
    return ABSTAIN


def lf_strong_good_segment_exact(row) -> int:
    """
    규칙 2: Segment 단위로 정확히 매칭되고(is_segment_exact_match == 1)
    BM25도 있으면 GOOD
    - Segment라도 정확성을 확인할 수 있다면 강한 신호 부여
    """
    if _get_row(row, TF.IS_SEGMENT_EXACT_MATCH, 0) == 1:
        if _get_row(row, TF.IS_BM25_MATCH, 0) == 1 or _get_row(row, TF.BM25_EXACT_RANK, 0) == 1:
            return GOOD
    
    return ABSTAIN


def lf_strong_bad_bm25_hybrid_conflict(row) -> int:
    """
    규칙 3: BM25에서 1위인데 Hybrid(vector포함)에서는 순위가 많이 밀리면 BAD
    - 키워드는 맞지만, 의미가 다를 가능성
    """
    if _get_row(row, TF.BM25_EXACT_RANK, 0) == 1 and _get_row(row, TF.BM25_TOP_RANK_IN_HYBRID) is not None and _get_row(row, TF.BM25_TOP_RANK_IN_HYBRID) >= 3:
        return BAD
    
    return ABSTAIN

# 해당 규칙은 ABSTAIN 감소 효과 목적으로 추가되었으며, 이후 지속적으로 사용할 것인지 판단 필요
def lf_good_rank1_with_exact(row) -> int:
    """
    규칙 3.5: Rank 1위이며, Exact match가 있으면 GOOD
    
    조건: 
    - is_exact_match == 1
    - bm25_top_rank_in_hybrid == 1
    
    이유: 
    - 정확한 매칭이 1위로 선택됨 = 가장 신뢰할 수 있는 Case
    - LLM fallback이어도 최종 선택은 exact match 기반
    """
    if (_get_row(row, TF.IS_EXACT_MATCH, 0) == 1 and
        _get_row(row, TF.BM25_TOP_RANK_IN_HYBRID) == 1):
        return GOOD
    
    return ABSTAIN


def lf_rank_mismatch_problem(row) -> int:
    """
    규칙 4: Rank 불일치 문제
    
    조건: exact match인데 rank가 3 이상 AND LLM fallback이 아님
    이유: 
    - DB에 정확한 매칭이 있는데 순위가 너무 낮음 = 랭킹 로직 문제
    - LLM fallback을 선택했다면 이미 검토된 결정이므로 문제될 것 없음
    """
    # LLM fallback이면 제외 (이미 의도된 선택)
    if _get_row(row, TF.IS_LLM_FALLBACK, 0) == 1:
        return ABSTAIN
    
    # 임계값을 3으로 완화 + reason이 exact_match인 경우만
    if (_get_row(row, TF.IS_EXACT_MATCH, 0) == 1 and 
        _get_row(row, TF.BM25_TOP_RANK_IN_HYBRID) is not None and 
        _get_row(row, TF.BM25_TOP_RANK_IN_HYBRID) >= 3 and
        _get_row(row, TF.REASON) == TR.EXACT_MATCH):
        return BAD
    
    return ABSTAIN


def lf_strong_bad_low_score_and_llm(row) -> int:
    """
    규칙 5: top_score가 낮고 LLM fallback이면 BAD
    - 검색 신호가 거의 없고 LLM으로 대체된 경우
    
    조건: LLM 사용 + toc_score < 0.017
    이유: 문장 번역이거나, Glossary에 없는 용어일 경우 LLM이 필수적이지만 번역결과 신뢰도를 확인 후 자동 저장 필요성 있음
    
    가드: exact_match가 있으면 정확한 매칭이 존재하므로 BAD 반환 금지
    """
    # 가드: exact_match가 있으면 BAD 반환 금지
    if _get_row(row, TF.IS_EXACT_MATCH, 0) == 1:
        return ABSTAIN
    
    if _get_row(row, TF.IS_LLM_FALLBACK, 0) == 1 and float(_get_row(row, TF.TOP_SCORE, 0)) < 0.017:
        return BAD
    
    return ABSTAIN


def lf_top_segment_but_rank_low(row) -> int:
    """
    규칙 6: Segment match인데 rank가 낮으면 BAD
    
    조건: is_top_segment = 1인데 hybrid_rank >= 2
    이유: 세그먼트 매칭은 있는데 1등이 아님 = 정확도 문제
    
    가드: exact_match가 있으면 정확한 매칭이 우선하므로 BAD 반환 금지
    """
    # 가드: exact_match가 있으면 BAD 반환 금지
    if _get_row(row, TF.IS_EXACT_MATCH, 0) == 1:
        return ABSTAIN
    
    if _get_row(row, TF.IS_TOP_SEGMENT, 0) == 1 and _get_row(row, TF.BM25_TOP_RANK_IN_HYBRID, 1) >= 2:
        return BAD
    return ABSTAIN


# ============================================================
# Labeling Functions (레이블링 규칙들)
# ============================================================
# 각 함수는 한 행(row)을 받아서 GOOD, BAD, ABSTAIN 중 하나를 반환

# 해당 규칙으로 inspect_weak_labels 결과 데이타가 오염되어 주석처리함.
# def lf_exact_match(row) -> int:
#     """
#     규칙 7: 정확 매칭이면 GOOD
    
#     이유: exact_match는 용어사전에 정확히 있다는 뜻
#     """
#     if row.get("is_exact_match"):
#         return GOOD
#     return ABSTAIN


def lf_bm25_match(row) -> int:
    """
    규칙 7: BM25에서 매칭되면 GOOD (키워드 일치)
    
    이유: BM25는 키워드 기반이므로 정확한 단어가 있다는 뜻
    """
    if row.get("is_bm25_match"):
        return GOOD
    return ABSTAIN


def lf_llm_fallback(row) -> int:
    """
    규칙 8: LLM fallback이면 BAD
    
    이유: 용어사전에 없어서 LLM으로 번역했다는 뜻
          (항상 나쁜 건 아니지만, 용어사전 매칭보다는 신뢰도 낮음)
    
    가드: exact_match가 있으면 LLM fallback이어도 정확한 매칭이 우선하므로 BAD 반환 금지
    """
    # 가드: exact_match가 있으면 BAD 반환 금지
    if _get_row(row, TF.IS_EXACT_MATCH, 0) == 1:
        return ABSTAIN
    
    if row.get("is_llm_fallback"):
        return BAD
    return ABSTAIN


def lf_high_score(row) -> int:
    """
    규칙 9: 점수가 높으면 GOOD
    
    이유: 검색 점수가 높다 = 의미적으로 유사하다
    임계값: 0.5 이상이면 꽤 높은 유사도
    """
    score = row.get("top_score")
    if score is not None and score >= 0.5:
        return GOOD
    return ABSTAIN


def lf_low_score(row) -> int:
    """
    규칙 10: 점수가 낮으면 BAD
    
    이유: 검색 점수가 낮다 = 제대로 된 매칭이 없다
    임계값: 0.1 미만이면 거의 관련 없는 결과
    
    가드: exact_match가 있으면 점수가 낮아도 정확한 매칭이 존재하므로 BAD 반환 금지
    """
    
    # 가드: exact_match가 있으면 BAD 반환 금지
    if _get_row(row, TF.IS_EXACT_MATCH, 0) == 1:
        return ABSTAIN
    
    if _get_row(row, TF.TOP_SCORE, 0) < 0.017:
        return BAD
    return ABSTAIN


def lf_strong_good_bm25_hybrid_agree(row) -> int:
    """
    규칙 11: BM25와 Hybrid 순위가 일치하면 GOOD
    
    이유: 키워드 검색과 의미 검색이 같은 결과 = 매우 신뢰할 수 있음
    """
    bm25_rank = row.get("bm25_exact_rank")
    hybrid_rank = row.get("bm25_top_rank_in_hybrid")
    
    # 둘 다 1위면 매우 신뢰
    if bm25_rank == 1 and hybrid_rank == 1:
        return GOOD
    return ABSTAIN


def lf_strong_bad_bm25_hybrid_disagree(row) -> int:
    """
    규칙 12: BM25에서 1위인데 Hybrid에서 순위가 많이 밀리면 BAD
    
    이유: 키워드는 맞는데 의미가 다르다 = 동음이의어 등 주의 필요
    """
    bm25_rank = row.get("bm25_exact_rank")
    hybrid_rank = row.get("bm25_top_rank_in_hybrid")
    
    if bm25_rank == 1 and hybrid_rank is not None and hybrid_rank >= 3:
        return BAD
    return ABSTAIN


# ============================================================
# 모든 Labeling Functions 리스트 (우선순위 순서)
# ============================================================
# 핵심 규칙: 클러스터 분석 기반 (가중치 높음)
CORE_LABELING_FUNCTIONS = [
    lf_strong_good_exact_bm25_no_llm,           # 가장 이상적
    lf_strong_good_segment_exact,               # exact + rank 1
    lf_strong_bad_bm25_hybrid_conflict,         # 개선 필요
    lf_good_rank1_with_exact,                   # rank 1 + exact 매칭
    lf_rank_mismatch_problem,                   # rank 불일치 큼
    lf_strong_bad_low_score_and_llm,            # segment 문제
    lf_top_segment_but_rank_low,         # 이상한 조합
]

# 보조 규칙: 기존 규칙들 (가중치 중간)
ASSISTANCE_LABELING_FUNCTIONS = [
    # lf_exact_match,
    lf_bm25_match,
    lf_llm_fallback,
    lf_strong_good_bm25_hybrid_agree,
    lf_strong_bad_bm25_hybrid_disagree,
]

# 추가 규칙: 약한 시그널 (가중치 낮음)
WEAK_LABELING_FUNCTIONS = [
    lf_high_score,
    lf_low_score,
]

# 전체 규칙 (순서 유지)
ALL_LABELING_FUNCTIONS = (
    CORE_LABELING_FUNCTIONS + 
    ASSISTANCE_LABELING_FUNCTIONS + 
    WEAK_LABELING_FUNCTIONS
)


# ============================================================
# 레이블 통합 (Majority Voting)
# ============================================================

def apply_labeling_functions(df: pd.DataFrame, lfs: list[Callable] = None) -> pd.DataFrame:
    """
    모든 Labeling Function을 데이터에 적용
    
    Args:
        df: 번역 로그 DataFrame
        lfs: 사용할 Labeling Functions 리스트 (기본값: 전체)
    
    Returns:
        각 LF의 결과를 컬럼으로 추가한 DataFrame
    """
    if lfs is None:
        lfs = ALL_LABELING_FUNCTIONS
    
    # 각 LF 적용
    for lf in lfs:
        col_name = lf.__name__  # 함수 이름을 컬럼명으로 사용
        df[col_name] = df.apply(lambda row: lf(row.to_dict()), axis=1)
    
    return df


def majority_vote(row, lf_columns: list[str]) -> int:
    """
    여러 Labeling Function의 결과를 다수결로 통합
    
    Args:
        row: DataFrame의 한 행
        lf_columns: Labeling Function 컬럼 이름들
    
    Returns:
        GOOD(1), BAD(0), 또는 ABSTAIN(-1)
    """
    votes = [row[col] for col in lf_columns if row[col] != ABSTAIN]
    
    if not votes:
        return ABSTAIN  # 모든 규칙이 판단 불가
    
    # 다수결
    good_count = sum(1 for v in votes if v == GOOD)
    bad_count = sum(1 for v in votes if v == BAD)
    
    if good_count > bad_count:
        return GOOD
    elif bad_count > good_count:
        return BAD
    else:
        return ABSTAIN  # 동점이면 판단 불가
    
    
def weighted_vote_with_meta(row, lf_columns: list[str]) -> tuple[int, dict]:
    """
    가중치 기반 투표 (클러스터 분석 기반 개선) + 메타 정보 반환
     
    핵심 규칙: 가중치 3.0
    보조 규칙: 가중치 1.5
    약한 규칙: 가중치 1.0
    
    Args:
        row: DataFrame의 한 행
        lf_columns: Labeling Function 컬럼 이름들
    
    Returns:
        GOOD(1), BAD(0), 또는 ABSTAIN(-1), 메타 정보
    """
    core_lf_names = [lf.__name__ for lf in CORE_LABELING_FUNCTIONS]
    assistance_lf_names = [lf.__name__ for lf in ASSISTANCE_LABELING_FUNCTIONS]
    
    good_score = 0.0
    bad_score = 0.0
    
    # label source 추적 용도
    best_good_rule = None
    best_good_weight = 0.0
    best_bad_rule = None
    best_bad_weight = 0.0
    
    saw_good = False
    saw_bad = False
    
    for col in lf_columns:
        vote = row[col]
        if vote == ABSTAIN:
            continue
        
        # 가중치 결정
        if col in core_lf_names:
            weight = 3.0
        elif col in assistance_lf_names:
            weight = 1.5
        else:
            weight = 1.0
            
        # 점수 누적 + 충돌 여부 체크
        if vote == GOOD:
            saw_good = True
            good_score += weight
            if weight > best_good_weight:
                best_good_weight = weight
                best_good_rule = col
                
        elif vote == BAD:
            saw_bad = True
            bad_score += weight
            if weight > best_bad_weight:
                best_bad_weight = weight
                best_bad_rule = col
            
    # 투표 자체가 없으면 ABSTAIN
    if good_score == 0 and bad_score == 0:
        return ABSTAIN, {
            "good_score": 0.0,
            "bad_score": 0.0,
            "label_strength": 0.0,
            "is_conflict": 0,
            "label_source": None,
        }
        
    elif good_score > bad_score:
        final = GOOD
    elif bad_score > good_score:
        final = BAD
    else:
        final = ABSTAIN
    
    # conflict: GOOD/BAD 규칙이 둘 다 있는 경우
    is_conflict = 1 if (saw_good and saw_bad) else 0
    
    # label_source: 최종 라벨쪽에서 "가장 높은 weight"로 한 표를 만든 규칙
    if final == GOOD:
        label_source = best_good_rule
    elif final == BAD:
        label_source = best_bad_rule
    else:
        label_source = None
    
    meta = {
        "good_score": float(good_score),
        "bad_score": float(bad_score),
        "label_strength": float(good_score - bad_score),
        "is_conflict": int(is_conflict),
        "label_source": label_source,
    }    
    return final, meta


def generate_weak_labels(df: pd.DataFrame, use_weighted_vote: bool = True) -> pd.DataFrame:
    """
    Weak Supervision으로 레이블 생성 (클러스터 분석 기반 개선)
    - use_weighted_vote=True:
        weak_label + (good_score, bad_score, label_strength, is_conflict, label_source)
    - use_weighted_vote=False:
        weak_label만 추가
    
    Args:
        df: 번역 로그 DataFrame
        use_weighted_vote: True면 가중치 기반, False면 단순 다수결
    
    Returns:
        weak_label 컬럼이 추가된 DataFrame
    
    사용법:
        df = translation_logger_db.get_logs_as_dataframe()
        df = generate_weak_labels(df, use_weighted_vote=True)
        print(df[['query', 'translation', 'weak_label']].head())
    """
    # 0. 파생 features 생성 (DB에 없는 것들)
    # is_top_segment: top_doc_type이 "cn_segment"이면 1
    if TF.TOP_DOC_TYPE in df.columns:
        df[TF.IS_TOP_SEGMENT] = (df[TF.TOP_DOC_TYPE] == "cn_segment").astype(int)
    else:
        df[TF.IS_TOP_SEGMENT] = 0
    
    # 1. 모든 Labeling Function 적용
    df = apply_labeling_functions(df)
    
    # 2. LF 컬럼 이름들
    lf_columns = [lf.__name__ for lf in ALL_LABELING_FUNCTIONS]
    
    # 3. 최종 레이블 생성 (가중치 or 단순 다수결)
    if use_weighted_vote:
        labels = []
        metas = []
        
        for idx, row in df.iterrows():
            final, meta = weighted_vote_with_meta(row, lf_columns)
            labels.append(final)
            metas.append(meta)
        
        df["weak_label"] = labels
        meta_df = pd.DataFrame(metas)
        df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
        vote_method = "가중치 기반 투표(+meta)"
        
    else:
        df["weak_label"] = df.apply(lambda row: majority_vote(row, lf_columns), axis=1)
        vote_method = "단순 다수결"
    
    # 4. 상세 통계 출력
    print(f"\n=== Weak Label 분포 ({vote_method}) ===")
    print(f"GOOD (1):     {(df['weak_label'] == GOOD).sum():4d} ({(df['weak_label'] == GOOD).sum()/len(df)*100:5.1f}%)")
    print(f"BAD (0):      {(df['weak_label'] == BAD).sum():4d} ({(df['weak_label'] == BAD).sum()/len(df)*100:5.1f}%)")
    print(f"ABSTAIN (-1): {(df['weak_label'] == ABSTAIN).sum():4d} ({(df['weak_label'] == ABSTAIN).sum()/len(df)*100:5.1f}%)")
    print(f"TOTAL:        {len(df):4d}")
    
    # 5. 규칙별 적용 통계 (ABSTAIN 제외)
    print(f"\n=== 규칙별 적용 통계 (상위 10개) ===")
    rule_stats = []
    for col in lf_columns:
        if col in df.columns:
            non_abstain = (df[col] != ABSTAIN).sum()
            good_votes = (df[col] == GOOD).sum()
            bad_votes = (df[col] == BAD).sum()
            rule_stats.append({
                'rule': col,
                'applied': non_abstain,
                'good': good_votes,
                'bad': bad_votes,
                'rate': non_abstain / len(df) * 100
            })
    
    rule_stats_df = pd.DataFrame(rule_stats).sort_values('applied', ascending=False)
    print(rule_stats_df.head(10).to_string(index=False))
    
    # 6. Conflict 통계
    if use_weighted_vote and "is_conflict" in df.columns:
        conflict_cnt = (df["is_conflict"] == 1).sum()
        print(f"\n=== Conflict 통계 ===")
        print(f"conflict rows: {conflict_cnt} ({conflict_cnt/len(df)*100:.1f}%)")
    
    # 7. Cluster별 레이블 분포 (cluster 컬럼이 있다면)
    if 'cluster' in df.columns:
        print(f"\n=== Cluster별 레이블 분포 ===")
        cluster_label = pd.crosstab(df['cluster'], df['weak_label'], margins=True)
        print(cluster_label)
    
    return df


# ============================================================
# 통합 분석: 클러스터링 + 레이블링
# ============================================================

def analyze_with_clustering(k: int = 6, use_weighted_vote: bool = True):
    """
    클러스터링과 Weak Supervision을 함께 실행하여 통합 분석
    
    Args:
        k: 클러스터 개수
        use_weighted_vote: 가중치 기반 투표 사용 여부
    
    Returns:
        클러스터와 레이블이 모두 포함된 DataFrame
    """
    import sys
    from pathlib import Path
    
    # 클러스터링 실행
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml_experiment.unsupervised_cluster_from_db import run_kmeans_clustering_from_db
    
    print("=" * 60)
    print("1단계: 비지도 클러스터링 실행")
    print("=" * 60)
    df = run_kmeans_clustering_from_db(k=k)
    
    print("\n" + "=" * 60)
    print("2단계: Weak Supervision 레이블링 실행")
    print("=" * 60)
    df = generate_weak_labels(df, use_weighted_vote=use_weighted_vote)
    
    return df


# ============================================================
# 테스트용 코드
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # 전체 분석 실행: python weak_supervision_labeler.py full
        df = analyze_with_clustering(k=6, use_weighted_vote=True)
        
        print("\n" + "=" * 60)
        print("분석 완료! 샘플 결과:")
        print("=" * 60)
        display_cols = ['query', 'translation', 'cluster', 'weak_label', 'reason']
        display_cols = [c for c in display_cols if c in df.columns]
        print(df[display_cols].head(20).to_string(index=False))
        
        # CSV 저장
        from pathlib import Path
        output_path = Path(__file__).parent / "labeled_logs.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {output_path}")
        
    else:
        # 테스트 데이터로 간단히 실행
        print("테스트 모드 (전체 분석: python weak_supervision_labeler.py full)")
        print("=" * 60)
        
        test_data = [
            {
                "query": "소원 운세 카드",
                "query_len": 8,
                "top_score": 0.8,
                "candidate_gap": 0.2,
                "is_exact_match": True,
                "is_bm25_match": True,
                "is_llm_fallback": False,
                "bm25_exact_rank": 1,
                "bm25_top_rank_in_hybrid": 1,
                "rank_diff": 0,
                "is_top_segment": 0,
            },
            {
                "query": "새로운용어",
                "query_len": 5,
                "top_score": 0.05,
                "candidate_gap": 0.001,
                "is_exact_match": False,
                "is_bm25_match": False,
                "is_llm_fallback": True,
                "bm25_exact_rank": 0,
                "bm25_top_rank_in_hybrid": 0,
                "rank_diff": 0,
                "is_top_segment": 0,
            },
            {
                "query": "애매한용어",
                "query_len": 5,
                "top_score": 0.3,
                "candidate_gap": 0.02,
                "is_exact_match": True,
                "is_bm25_match": True,
                "is_llm_fallback": True,
                "bm25_exact_rank": 1,
                "bm25_top_rank_in_hybrid": 3,
                "rank_diff": 2,
                "is_top_segment": 0,
            },
            {
                "query": "【海外+7】鹿游角色单",
                "query_len": 11,
                "top_score": 0.017,
                "candidate_gap": 0.0003,
                "is_exact_match": True,
                "is_bm25_match": False,
                "is_llm_fallback": True,
                "bm25_exact_rank": 0,
                "bm25_top_rank_in_hybrid": 1,
                "rank_diff": 1,
                "is_top_segment": 0,
            },
            {
                "query": "提升同步等级",
                "query_len": 6,
                "top_score": 0.033,
                "candidate_gap": 0.017,
                "is_exact_match": True,
                "is_bm25_match": True,
                "is_llm_fallback": False,
                "bm25_exact_rank": 1,
                "bm25_top_rank_in_hybrid": 1,
                "rank_diff": 0,
                "is_top_segment": 0,
            },
        ]
        
        df = pd.DataFrame(test_data)
        df = generate_weak_labels(df, use_weighted_vote=True)
        
        print("\n=== 테스트 결과 ===")
        print(df[["query", "weak_label", "is_exact_match", "is_bm25_match", "is_llm_fallback"]].to_string(index=False))

