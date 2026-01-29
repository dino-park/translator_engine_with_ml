import pandas as pd
import numpy as np
import sys

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 상위 디렉토리를 path에 추가 (translation_logger_db 임포트용)
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_logs_from_db() -> pd.DataFrame:
    """
    TranslationLoggerDB에서 번역 로그를 로드하여 DataFrame으로 반환

    Returns:
        pd.DataFrame: 번역 로그 DataFrame
    """
    
    from translation_logger_db import translation_logger_db
    
    df = translation_logger_db.get_logs_as_dataframe()
    return df


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    ML clustering에 사용할 feature 정리.
    - 결측치 처리
    - bool/int 반환
    - 필요 시, 파생 feature(rank_diff) 추가
    """
    
    df = df.copy()
    
    # 1. 기본 feature 후보
    feature_cols = [
        "query_len",
        "top_score",
        "candidate_gap",
        "is_exact_match",
        "is_segment_exact_match",
        "is_bm25_match",
        "is_llm_fallback",
    ]
    
    # 2. 선택적 feature (DB에 존재할 경우만 사용)
    optional_cols = ["bm25_exact_rank", "bm25_top_rank_in_hybrid"]
    for col in optional_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    # 3. rank_diff 파생 feature (둘 다 있을 경우에만, hybrid와 BM25 순위 차이)
    if "bm25_exact_rank" in df.columns and "bm25_top_rank_in_hybrid" in df.columns:
        df["rank_diff"] = (df["bm25_top_rank_in_hybrid"].fillna(0) - df["bm25_exact_rank"].fillna(0))
        feature_cols.append("rank_diff")
        
    if "top_doc_type" in df.columns:
        df["is_top_segment"] = (df["top_doc_type"].fillna("") == "cn_segment").astype(int)
        feature_cols.append("is_top_segment")
    
    # 4. 결측치 처리: 숫자는 0, bool은 0/1로    
    numeric_cols = [
        "query_len", "top_score", "candidate_gap",
        "bm25_exact_rank", "bm25_top_rank_in_hybrid", "rank_diff"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    bool_cols = ["is_exact_match", "is_bm25_match", "is_llm_fallback", "is_segment_exact_match"]
    for col in bool_cols:
        if col in df.columns:
            # SQLite에서 0/1로 들어와도, 혹시 True/False로 들어와도 안전하게 int로 처리
            df[col] = df[col].fillna(0).astype(int)
            
    if "query_len" not in df.columns:
        df["query_len"] = df["query"].fillna("").apply(len)
        if "query_len" not in feature_cols:
            feature_cols.insert(0, "query_len")
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


def run_kmeans_clustering_from_db(k: int = 6, min_rows: int = 100) -> pd.DataFrame:
    
    df = load_logs_from_db()
    
    if df.empty:
        raise ValueError("DB에 저장된 로그가 0개입니다. 먼저 translation_logger_db.log_translation()으로 로그를 쌓아주세요 ")
        return df
    
    df, feature_cols = preprocess_features(df)
    
    # 데이타 갯수가 너무 적으면 Clustering이 의가 없거나 불안정함
    if len(df) < min_rows:
        print(f"로그가 {len(df)}로, 비지도 클러스터링이 불안정할 수 있음. (권장: {min_rows}개 이상)")
        print("진행합니다.\n")
    
    if len(feature_cols) == 0:
        print("clustering에 사용할 feature col이 없습니다. ")
        return df
    
    # Cluster 갯수 k는 데이타 갯수보다 많을 수 없음
    k = min(k, len(df))
    if k < 2:
        print("클러스터링을 하려면 최소 K는 2이상 이어야 합니다. ")
        return df
    
    # X 데이타 준비
    # 번역로그(df)에서 숫자 feature 컬럼만 추출, 숫자 크기 차이를 맞춘 다음(스케일링)
    # 비슷한 패턴끼리 자동으로 그룹(cluster)을 나눔(KMeans)
    X = df[feature_cols].values
    # 데이타 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    # n_clusters: Cluster 그룹을 k개로 나눔(로그 패턴을 6개 유형으로 나눔)
    # random_state: 랜덤 시드 고정(동일한 데이터에 대해 동일한 결과 보장)
    # n_init: KMeans 알고리즘 반복 횟수(초기 클러스터 중심 초기화 방법을 10번 시도)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X_scaled)
    
    # 결과 요약
    print("\n=== 사용 feature columns ===")
    print(feature_cols)
    
    print("\n=== Cluster 요약(각 그룹 평균) ===")
    summary = df.groupby("cluster")[feature_cols].mean().round(4)
    print(summary)
    
    print("\n=== 각 Cluster별 샘플(각 그룹 5개) ===")
    show_cols = [c for c in ["created_at", "query", "translation", "reason"] if c in df.columns]
    show_cols += feature_cols
    
    for c in sorted(df["cluster"].unique()):
        print(f"\n[cluster={c}] count={len(df[df['cluster'] == c])}")
        print(df[df['cluster'] == c][show_cols].head(5))
        
    save_csv = Path(__file__).parent / "clustered_logs.csv"
    df.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"\n클러스터 결과를 csv로 저장함: {save_csv}")
    return df
    

if __name__ == "__main__":
    run_kmeans_clustering_from_db(k=6)

    