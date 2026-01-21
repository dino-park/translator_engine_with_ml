"""
Glossary 매칭 신뢰도 예측 모델 학습

목적:
- "이 번역 결과를 신뢰할 수 있는가?" 예측
- exact_match 아니지만 실제로 맞는 경우 찾기
- LLM 호출이 불필요한 경우 감지

사용 모델: Logistic Regression (scikit-learn)
- 이진 분류: GOOD(1) vs BAD(0)
- 해석 가능: 어떤 피처가 중요한지 알 수 있음
- 빠름: 학습/추론 모두 빠름

실행 방법:
    python train_confidence_model.py
"""

import sys
from pathlib import Path

# 상위 디렉토리를 path에 추가 (translation_logger_db 임포트용)
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib  # 모델 저장용

# Weak Supervision 레이블러 임포트
from weak_supervision_labeler import generate_weak_labels, GOOD, BAD, ABSTAIN


# ============================================================
# 1단계: 데이터 로드
# ============================================================

def load_data_from_db():
    """
    SQLite DB에서 번역 로그 로드
    """
    from translation_logger_db import translation_logger_db
    
    df = translation_logger_db.get_logs_as_dataframe()
    print(f"로드된 데이터: {len(df)}개")
    return df


def load_data_from_csv(csv_path: str):
    """
    CSV 파일에서 데이터 로드 (DB 없을 때 테스트용)
    """
    df = pd.read_csv(csv_path)
    print(f"로드된 데이터: {len(df)}개")
    return df


# ============================================================
# 2단계: 피처 엔지니어링
# ============================================================

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    ML 모델에 사용할 피처 준비
    
    Args:
        df: 원본 DataFrame
    
    Returns:
        (피처가 정리된 DataFrame, 피처 컬럼 이름 리스트)
    """
    # 사용할 피처들
    feature_columns = [
        "query_len",        # 쿼리 길이
        "top_score",        # 검색 1위 점수
        "candidate_gap",    # 1위-2위 점수 차이
        "is_exact_match",   # 정확 매칭 여부
        "is_bm25_match",    # BM25 매칭 여부
        "is_llm_fallback",  # LLM fallback 여부
    ]
    
    # 결측치 처리
    df = df.copy()
    
    # 숫자형 컬럼: 0으로 채움
    for col in ["top_score", "candidate_gap", "bm25_exact_rank", "bm25_hybrid_rank"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Boolean 컬럼: False로 채움
    for col in ["is_exact_match", "is_bm25_match", "is_llm_fallback"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
    
    # BM25 분석 피처 추가 (있으면)
    if "bm25_exact_rank" in df.columns:
        feature_columns.append("bm25_exact_rank")
    if "bm25_hybrid_rank" in df.columns:
        feature_columns.append("bm25_hybrid_rank")
    
    # 파생 피처: BM25와 Hybrid 순위 차이
    if "bm25_exact_rank" in df.columns and "bm25_hybrid_rank" in df.columns:
        df["rank_diff"] = (df["bm25_hybrid_rank"] - df["bm25_exact_rank"]).fillna(0)
        feature_columns.append("rank_diff")
    
    print(f"사용할 피처: {feature_columns}")
    return df, feature_columns


# ============================================================
# 3단계: 학습 데이터 준비
# ============================================================

def prepare_training_data(df: pd.DataFrame, feature_columns: list[str]):
    """
    학습용 X, y 데이터 준비
    
    Args:
        df: Weak Label이 포함된 DataFrame
        feature_columns: 사용할 피처 컬럼들
    
    Returns:
        X, y (ABSTAIN 제외)
    """
    # ABSTAIN(-1) 제외하고 GOOD(1), BAD(0)만 사용
    df_labeled = df[df["weak_label"] != ABSTAIN].copy()
    print(f"레이블이 있는 데이터: {len(df_labeled)}개 (ABSTAIN 제외)")
    
    # X: 피처, y: 레이블
    X = df_labeled[feature_columns].values
    y = df_labeled["weak_label"].values
    
    return X, y, df_labeled


# ============================================================
# 4단계: 모델 학습
# ============================================================

def train_model(X, y, test_size: float = 0.2):
    """
    Logistic Regression 모델 학습
    
    Args:
        X: 피처 배열
        y: 레이블 배열
        test_size: 테스트 데이터 비율
    
    Returns:
        (학습된 모델, scaler, 테스트 데이터)
    """
    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
    
    # 피처 스케일링 (Logistic Regression은 스케일링하면 성능 향상)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,  # 수렴 보장
        class_weight="balanced"  # 클래스 불균형 처리
    )
    model.fit(X_train_scaled, y_train)
    
    # 테스트 정확도
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"\n학습 정확도: {train_acc:.4f}")
    print(f"테스트 정확도: {test_acc:.4f}")
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    
    # 상세 평가
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["BAD", "GOOD"]))
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler, (X_test_scaled, y_test, y_pred)


# ============================================================
# 5단계: 피처 중요도 분석
# ============================================================

def analyze_feature_importance(model, feature_columns: list[str]):
    """
    어떤 피처가 중요한지 분석
    
    Args:
        model: 학습된 Logistic Regression 모델
        feature_columns: 피처 이름 리스트
    """
    # 계수 = 피처 중요도 (절대값이 클수록 중요)
    coefficients = model.coef_[0]
    
    # 중요도 순으로 정렬
    importance = sorted(
        zip(feature_columns, coefficients),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    print("\n=== 피처 중요도 (Coefficients) ===")
    print("양수: GOOD에 기여 / 음수: BAD에 기여")
    print("-" * 40)
    for name, coef in importance:
        direction = "↑ GOOD" if coef > 0 else "↓ BAD"
        print(f"{name:20s}: {coef:+.4f} ({direction})")


# ============================================================
# 6단계: 모델 저장/로드
# ============================================================

def save_model(model, scaler, feature_columns: list[str], save_dir: str = "model"):
    """
    학습된 모델 저장
    """
    save_path = Path(__file__).parent / save_dir
    save_path.mkdir(exist_ok=True)
    
    joblib.dump(model, save_path / "confidence_model.pkl")
    joblib.dump(scaler, save_path / "scaler.pkl")
    joblib.dump(feature_columns, save_path / "feature_columns.pkl")
    
    print(f"\n모델 저장 완료: {save_path}")


def load_model(model_dir: str = "model"):
    """
    저장된 모델 로드
    """
    load_path = Path(__file__).parent / model_dir
    
    model = joblib.load(load_path / "confidence_model.pkl")
    scaler = joblib.load(load_path / "scaler.pkl")
    feature_columns = joblib.load(load_path / "feature_columns.pkl")
    
    return model, scaler, feature_columns


# ============================================================
# 7단계: 예측 함수 (실제 사용용)
# ============================================================

def predict_confidence(result: dict, model=None, scaler=None, feature_columns=None) -> float:
    """
    번역 결과의 신뢰도 예측
    
    Args:
        result: translate_term()의 반환값
        model, scaler, feature_columns: 학습된 모델 (없으면 로드)
    
    Returns:
        신뢰도 확률 (0.0 ~ 1.0)
    
    사용법:
        result = translate_term(query, hybrid, src_lang)
        confidence = predict_confidence(result)
        if confidence > 0.7:
            print("용어사전 결과 사용")
        else:
            print("LLM 검토 필요")
    """
    if model is None:
        model, scaler, feature_columns = load_model()
    
    # 피처 추출
    features = []
    for col in feature_columns:
        if col == "query_len":
            features.append(len(result.get("query", "")))
        elif col == "top_score":
            candidates = result.get("candidates", [])
            features.append(candidates[0]["score"] if candidates else 0)
        elif col == "candidate_gap":
            candidates = result.get("candidates", [])
            gap = (candidates[0]["score"] - candidates[1]["score"]) if len(candidates) > 1 else 0
            features.append(gap)
        elif col == "is_exact_match":
            features.append(1 if "exact_match" in result.get("reason", "") else 0)
        elif col == "is_bm25_match":
            bm25 = result.get("bm25_analysis") or {}
            features.append(1 if bm25.get("bm25_exact_rank") else 0)
        elif col == "is_llm_fallback":
            features.append(1 if "llm" in result.get("reason", "").lower() else 0)
        elif col == "bm25_exact_rank":
            bm25 = result.get("bm25_analysis") or {}
            features.append(bm25.get("bm25_exact_rank") or 0)
        elif col == "bm25_hybrid_rank":
            bm25 = result.get("bm25_analysis") or {}
            features.append(bm25.get("bm25_hybrid_rank") or 0)
        elif col == "rank_diff":
            bm25 = result.get("bm25_analysis") or {}
            exact = bm25.get("bm25_exact_rank") or 0
            hybrid = bm25.get("bm25_hybrid_rank") or 0
            features.append(hybrid - exact)
        else:
            features.append(0)
    
    # 스케일링 + 예측
    X = np.array([features])
    X_scaled = scaler.transform(X)
    
    # 확률 예측 (GOOD일 확률)
    proba = model.predict_proba(X_scaled)[0][1]
    
    return proba


# ============================================================
# 메인 실행
# ============================================================

def main():
    """
    전체 학습 파이프라인 실행
    """
    print("=" * 50)
    print("Glossary 매칭 신뢰도 예측 모델 학습")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드")
    try:
        df = load_data_from_db()
    except Exception as e:
        print(f"DB 로드 실패: {e}")
        print("테스트 데이터로 진행합니다.")
        df = create_test_data()
    
    # 2. 피처 준비
    print("\n[2단계] 피처 엔지니어링")
    df, feature_columns = prepare_features(df)
    
    # 3. Weak Label 생성
    print("\n[3단계] Weak Supervision 레이블 생성")
    df = generate_weak_labels(df)
    
    # 4. 학습 데이터 준비
    print("\n[4단계] 학습 데이터 준비")
    X, y, df_labeled = prepare_training_data(df, feature_columns)
    
    if len(X) < 10:
        print("데이터가 너무 적습니다 (최소 10개 필요)")
        return
    
    # 5. 모델 학습
    print("\n[5단계] 모델 학습")
    model, scaler, test_data = train_model(X, y)
    
    # 6. 피처 중요도 분석
    analyze_feature_importance(model, feature_columns)
    
    # 7. 모델 저장
    print("\n[6단계] 모델 저장")
    save_model(model, scaler, feature_columns)
    
    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)


def create_test_data() -> pd.DataFrame:
    """
    DB가 없을 때 사용할 테스트 데이터 생성
    """
    np.random.seed(42)
    n = 100  # 샘플 수
    
    data = {
        "query": [f"테스트용어{i}" for i in range(n)],
        "query_len": np.random.randint(2, 15, n),
        "top_score": np.random.uniform(0, 1, n),
        "candidate_gap": np.random.uniform(0, 0.3, n),
        "is_exact_match": np.random.choice([True, False], n, p=[0.3, 0.7]),
        "is_bm25_match": np.random.choice([True, False], n, p=[0.4, 0.6]),
        "is_llm_fallback": np.random.choice([True, False], n, p=[0.3, 0.7]),
        "bm25_exact_rank": np.random.choice([1, 2, 3, None], n),
        "bm25_hybrid_rank": np.random.choice([1, 2, 3, 4, 5, None], n),
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()

