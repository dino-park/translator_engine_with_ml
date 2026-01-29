# inspect_weak_labels.py
"""
Weak Label 품질 점검 스크립트
실행:
    python -m ml_experiment.inspect_weak_labels

옵션(전체 Pipline: Clustering + Weak Labeling)
    python -m ml_experiment.inspect_weak_labels full
"""
import sys
import pandas as pd
from pathlib import Path
# 상위 디렉토리를 path에 추가 (translation_logger_db 임포트용)
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_experiment.weak_supervision_labeler import (
    GOOD, BAD, ABSTAIN,
    ALL_LABELING_FUNCTIONS,
    generate_weak_labels,
)
from ml_experiment.unsupervised_cluster_from_db import load_logs_from_db
from core.config import TranslationFeatures as TF


def load_data_from_db() -> pd.DataFrame:
    from translation_logger_db import translation_logger_db
    return translation_logger_db.get_logs_as_dataframe()


def print_basic_distribution(df: pd.DataFrame):
    total = len(df)
    good = (df['weak_label'] == GOOD).sum()
    bad = (df['weak_label'] == BAD).sum()
    abstain = (df['weak_label'] == ABSTAIN).sum()
    
    print("\n" + "=" * 60)
    print("[1] 레이블 분포 / 커버리지")
    print("=" * 60)
    print(f"TOTAL: {total}")
    print(f"GOOD (1): {good} ({good/total*100:.1f}%)")
    print(f"BAD (0): {bad} ({bad/total*100:.1f}%)")
    print(f"ABSTAIN (-1): {abstain} ({abstain/total*100:.1f}%)")
    

def print_conflict_strength_stats(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[2] 충돌(conflict) / 강도(label_strength) 통계")
    print("=" * 60)
    
    if "is_conflict" not in df.columns or "label_strength" not in df.columns:
        print("df에 is_conflict 또는 label_strength 컬럼이 없습니다.")
        print("먼저 genereate_weak_labels()에서 meta컬럼 저장을 적용하세요.")
        return
    
    conflict_cnt = (df["is_conflict"] == 1).sum()
    print(f"Conflict rows: {conflict_cnt} ({conflict_cnt/len(df)*100:.1f}%)")
    
    # label_strength: GOOD이면 양수, BAD면 음수, ABSTAIN은 0에 가까움
    print("\nlabel_strength 요약 (전체):")
    print(df["label_strength"].describe().round(4))
    
    # "애매한 레이블" 후보: label_strength가 작은 경우
    ambiguous = df[df["weak_label"] != ABSTAIN].copy()
    ambiguous["abs_strength"] = ambiguous["label_strength"].abs()
    if not ambiguous.empty:
        print("\n|label_strength|가 작은 Top 10(애매한 케이스):")
        # TODO: TranslationFeatures 클래스에 weak label 컬럼 추가 후 이 부분 수정
        cols = [c for c in [TF.QUERY, TF.TRANSLATION, TF.REASON, TF.WEAK_LABEL, TF.LABEL_STRENGTH, TF.IS_CONFLICT, TF.LABEL_SOURCE] if c in df.columns]
        try:
            print(ambiguous.sort_values("abs_strength", ascending=True)[cols].head(10).to_string(index=False))
        except UnicodeEncodeError:
            print("(터미널 인코딩 제한으로 출력 생략. CSV 파일을 확인하세요.)")
    

def print_rule_execution_stats(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[3] 규칙별 적용 통계")
    print("=" * 60)
    
    lf_names = [lf.__name__ for lf in ALL_LABELING_FUNCTIONS]
    stats = []
    
    for name in lf_names:
        if name not in df.columns:
            continue
        applied = (df[name] != ABSTAIN).sum()
        good_votes = (df[name] == GOOD).sum()
        bad_votes = (df[name] == BAD).sum()
        stats.append({
            "rule": name,
            "applied": applied,
            "applied_rate_%": round(applied/len(df)*100, 2),
            "good_votes": good_votes,
            "bad_votes": bad_votes,
        })
        
    if not stats:
        print("규칙 컬럼이 df에 없습니다. (apply_labeling_functions가 실행됐는지 확인)")
        return
    
    out = pd.DataFrame(stats).sort_values("applied", ascending=False)
    try:
        print(out.head(15).to_string(index=False))
    except UnicodeEncodeError:
        print("(터미널 인코딩 제한으로 출력 생략. CSV 파일을 확인하세요.)")
    
    print("\n✅ 해석 팁:")
    print("- applied_rate_%가 너무 높은 규칙 1~2개가 전체를 지배하면, 라벨이 단조로워질 수 있어요.")
    print("- bad_votes가 과도한 규칙은 '과격한 BAD 규칙'일 수 있으니 threshold를 점검해보세요.")


def print_cluster_label_distribution(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[4] Cluster별 라벨 분포 (cluster 컬럼이 있을 때만)")
    print("=" * 60)

    if "cluster" not in df.columns:
        print("cluster 컬럼이 없습니다. (클러스터링까지 같이 보고 싶으면 full 모드로 실행하세요)")
        return

    tab = pd.crosstab(df["cluster"], df["weak_label"], margins=True)
    print(tab)

    print("\n✅ 해석 팁:")
    print("- 특정 cluster가 GOOD/BAD 한쪽으로 거의 쏠리면, 그 cluster는 '성격이 뚜렷'하다고 볼 수 있어요.")
    print("- cluster 내부에서 ABST가 많으면: 그 cluster를 규칙이 잘 못 커버하는 상태일 수 있어요.")


def export_review_samples(df: pd.DataFrame, out_path: str = "weak_label_review_samples.csv"):
    """
    사람이 빠르게 검토할 샘플을 저장:
    - conflict 케이스
    - strength가 낮은 애매 케이스
    - strength가 높은 확신 케이스(GOOD/BAD 각각)
    """
    cols = [c for c in [TF.CREATED_AT, TF.QUERY, TF.TRANSLATION, TF.REASON, TF.WEAK_LABEL, TF.LABEL_SOURCE, TF.LABEL_STRENGTH, TF.IS_CONFLICT] if c in df.columns]
    if not cols:
        cols = df.columns.tolist()

    samples = []

    # 1) conflict 샘플
    if "is_conflict" in df.columns:
        samples.append(df[df["is_conflict"] == 1].head(50))

    # 2) 애매한 샘플(|strength| 낮음)
    if "label_strength" in df.columns:
        tmp = df[df["weak_label"] != ABSTAIN].copy()
        tmp["abs_strength"] = tmp["label_strength"].abs()
        samples.append(tmp.sort_values("abs_strength", ascending=True).head(50))

        # 3) 확신 GOOD/BAD 샘플
        samples.append(tmp.sort_values("abs_strength", ascending=False).head(50))

    out = pd.concat(samples, axis=0)
    # query + translation 기준으로 중복 제거 (같은 row가 여러 샘플 그룹에 포함될 수 있음)
    if TF.QUERY in out.columns and TF.TRANSLATION in out.columns:
        out = out.drop_duplicates(subset=[TF.QUERY, TF.TRANSLATION], keep='first')
    else:
        # 모든 데이터 컬럼 기준 중복 제거
        out = out.drop_duplicates(subset=[c for c in cols if c in out.columns], keep='first')
    out = out[cols].reset_index(drop=True)  # 필요한 컬럼만 선택하고 인덱스 초기화
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[5] 리뷰 샘플 CSV 저장: {out_path}")
    print("✅ 이 파일을 직접 확인하며 '규칙이 맞는지' 검토하면, 규칙 개선이 빨라집니다.")


def run(full: bool = False):
    if full:
        # 클러스터링까지 같이 돌리고 싶으면 weak_supervision_labeler의 통합 함수를 써도 됨
        from ml_experiment.weak_supervision_labeler import analyze_with_clustering
        df = analyze_with_clustering(k=6, use_weighted_vote=True)
    else:
        df = load_logs_from_db()
        if df.empty:
            print("DB 로그가 없습니다. 먼저 번역 요청을 쌓아주세요.")
            return
        df = generate_weak_labels(df, use_weighted_vote=True)

    print_basic_distribution(df)
    print_conflict_strength_stats(df)
    print_rule_execution_stats(df)
    print_cluster_label_distribution(df)
    export_review_samples(df)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    run(full=(arg == "full"))