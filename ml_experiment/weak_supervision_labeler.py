"""
Weak Supervision 레이블러

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

import pandas as pd
import numpy as np
from typing import Callable

# 레이블 상수 정의
GOOD = 1      # 신뢰할 수 있음
BAD = 0       # 신뢰하기 어려움
ABSTAIN = -1  # 판단 불가


# ============================================================
# Labeling Functions (레이블링 규칙들)
# ============================================================
# 각 함수는 한 행(row)을 받아서 GOOD, BAD, ABSTAIN 중 하나를 반환

def lf_exact_match(row) -> int:
    """
    규칙 1: 정확 매칭이면 GOOD
    
    이유: exact_match는 용어사전에 정확히 있다는 뜻
    """
    if row.get("is_exact_match"):
        return GOOD
    return ABSTAIN


def lf_bm25_match(row) -> int:
    """
    규칙 2: BM25에서 매칭되면 GOOD (키워드 일치)
    
    이유: BM25는 키워드 기반이므로 정확한 단어가 있다는 뜻
    """
    if row.get("is_bm25_match"):
        return GOOD
    return ABSTAIN


def lf_llm_fallback(row) -> int:
    """
    규칙 3: LLM fallback이면 BAD
    
    이유: 용어사전에 없어서 LLM으로 번역했다는 뜻
          (항상 나쁜 건 아니지만, 용어사전 매칭보다는 신뢰도 낮음)
    """
    if row.get("is_llm_fallback"):
        return BAD
    return ABSTAIN


def lf_high_score(row) -> int:
    """
    규칙 4: 점수가 높으면 GOOD
    
    이유: 검색 점수가 높다 = 의미적으로 유사하다
    임계값: 0.5 이상이면 꽤 높은 유사도
    """
    score = row.get("top_score")
    if score is not None and score >= 0.5:
        return GOOD
    return ABSTAIN


def lf_low_score(row) -> int:
    """
    규칙 5: 점수가 낮으면 BAD
    
    이유: 검색 점수가 낮다 = 제대로 된 매칭이 없다
    임계값: 0.1 미만이면 거의 관련 없는 결과
    """
    score = row.get("top_score")
    if score is not None and score < 0.1:
        return BAD
    return ABSTAIN


def lf_large_gap(row) -> int:
    """
    규칙 6: 1위-2위 점수 차이가 크면 GOOD
    
    이유: 점수 차이가 크다 = 1위가 확실히 맞다
    임계값: 0.1 이상 차이면 확실한 1위
    """
    gap = row.get("candidate_gap")
    if gap is not None and gap >= 0.1:
        return GOOD
    return ABSTAIN


def lf_small_gap(row) -> int:
    """
    규칙 7: 1위-2위 점수 차이가 작으면 BAD
    
    이유: 점수 차이가 작다 = 여러 후보 중 어떤 게 맞는지 애매함
    임계값: 0.01 미만이면 거의 동점
    """
    gap = row.get("candidate_gap")
    if gap is not None and gap < 0.01:
        return BAD
    return ABSTAIN


def lf_short_query(row) -> int:
    """
    규칙 8: 짧은 쿼리 + exact_match면 GOOD
    
    이유: 짧은 용어는 정확 매칭이 더 신뢰할 수 있음
    """
    query_len = row.get("query_len", 0)
    is_exact = row.get("is_exact_match", False)
    
    if query_len <= 5 and is_exact:
        return GOOD
    return ABSTAIN


def lf_bm25_and_hybrid_agree(row) -> int:
    """
    규칙 9: BM25와 Hybrid 순위가 일치하면 GOOD
    
    이유: 키워드 검색과 의미 검색이 같은 결과 = 매우 신뢰할 수 있음
    """
    bm25_rank = row.get("bm25_exact_rank")
    hybrid_rank = row.get("bm25_hybrid_rank")
    
    # 둘 다 1위면 매우 신뢰
    if bm25_rank == 1 and hybrid_rank == 1:
        return GOOD
    return ABSTAIN


def lf_bm25_and_hybrid_disagree(row) -> int:
    """
    규칙 10: BM25에서 1위인데 Hybrid에서 순위가 많이 밀리면 BAD
    
    이유: 키워드는 맞는데 의미가 다르다 = 동음이의어 등 주의 필요
    """
    bm25_rank = row.get("bm25_exact_rank")
    hybrid_rank = row.get("bm25_hybrid_rank")
    
    if bm25_rank == 1 and hybrid_rank is not None and hybrid_rank >= 3:
        return BAD
    return ABSTAIN


# ============================================================
# 모든 Labeling Functions 리스트
# ============================================================
ALL_LABELING_FUNCTIONS = [
    lf_exact_match,
    lf_bm25_match,
    lf_llm_fallback,
    lf_high_score,
    lf_low_score,
    lf_large_gap,
    lf_small_gap,
    lf_short_query,
    lf_bm25_and_hybrid_agree,
    lf_bm25_and_hybrid_disagree,
]


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


def generate_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weak Supervision으로 레이블 생성
    
    Args:
        df: 번역 로그 DataFrame
    
    Returns:
        weak_label 컬럼이 추가된 DataFrame
    
    사용법:
        df = translation_logger_db.get_logs_as_dataframe()
        df = generate_weak_labels(df)
        print(df[['query', 'weak_label']].head())
    """
    # 1. 모든 Labeling Function 적용
    df = apply_labeling_functions(df)
    
    # 2. LF 컬럼 이름들
    lf_columns = [lf.__name__ for lf in ALL_LABELING_FUNCTIONS]
    
    # 3. 다수결로 최종 레이블 생성
    df["weak_label"] = df.apply(lambda row: majority_vote(row, lf_columns), axis=1)
    
    # 4. 레이블 통계 출력
    print("=== Weak Label 분포 ===")
    print(f"GOOD (1):    {(df['weak_label'] == GOOD).sum()}")
    print(f"BAD (0):     {(df['weak_label'] == BAD).sum()}")
    print(f"ABSTAIN (-1): {(df['weak_label'] == ABSTAIN).sum()}")
    
    return df


# ============================================================
# 테스트용 코드
# ============================================================

if __name__ == "__main__":
    # 테스트 데이터 생성
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
            "bm25_hybrid_rank": 1,
        },
        {
            "query": "새로운용어",
            "query_len": 5,
            "top_score": 0.05,
            "candidate_gap": 0.001,
            "is_exact_match": False,
            "is_bm25_match": False,
            "is_llm_fallback": True,
            "bm25_exact_rank": None,
            "bm25_hybrid_rank": None,
        },
        {
            "query": "애매한용어",
            "query_len": 5,
            "top_score": 0.3,
            "candidate_gap": 0.02,
            "is_exact_match": False,
            "is_bm25_match": True,
            "is_llm_fallback": False,
            "bm25_exact_rank": 2,
            "bm25_hybrid_rank": 3,
        },
    ]
    
    df = pd.DataFrame(test_data)
    df = generate_weak_labels(df)
    
    print("\n=== 결과 ===")
    print(df[["query", "weak_label"]])

