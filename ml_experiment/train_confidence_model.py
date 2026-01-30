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

# Config에서 feature 정의 임포트
from core.config import TranslationFeatures, ML_FEATURE_COLUMNS, ML_OPTIONAL_FEATURES


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
    df = df.copy()
    
    # config.py에서 정의한 feature 리스트 사용
    feature_columns = ML_FEATURE_COLUMNS.copy()
    
    # Boolean 컬럼을 int로 변환 (ML 모델용)
    bool_features = [
        TranslationFeatures.IS_EXACT_MATCH,
        TranslationFeatures.IS_BM25_MATCH,
        TranslationFeatures.IS_LLM_FALLBACK,
        TranslationFeatures.IS_SEGMENT_EXACT_MATCH,
        TranslationFeatures.IS_TOP_SEGMENT,
    ]
    for col in bool_features:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Optional features 추가 (데이터에 있으면)
    for col in ML_OPTIONAL_FEATURES:
        if col == TranslationFeatures.RANK_DIFF:
            # 파생 피처: BM25와 Hybrid 순위 차이
            bm25_exact = TranslationFeatures.BM25_EXACT_RANK
            bm25_hybrid = TranslationFeatures.BM25_TOP_RANK_IN_HYBRID
            
            if bm25_exact in df.columns and bm25_hybrid in df.columns:
                df[col] = df[bm25_hybrid] - df[bm25_exact]
                df[col] = df[col].fillna(0)  # 순위 차이가 없으면 0
                feature_columns.append(col)
        elif col in df.columns:
            # 기존 optional feature
            feature_columns.append(col)
    
    print(f"사용할 피처: {feature_columns}")
    print(f"  - 기본 피처: {len(ML_FEATURE_COLUMNS)}개")
    print(f"  - 추가 피처: {len(feature_columns) - len(ML_FEATURE_COLUMNS)}개")
    
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
    
    # 피처 추출 (TranslationFeatures 사용)
    features = []
    for col in feature_columns:
        if col == TranslationFeatures.QUERY_LEN:
            features.append(len(result.get(TranslationFeatures.QUERY, "")))
        elif col == TranslationFeatures.TOP_SCORE:
            candidates = result.get(TranslationFeatures.CANDIDATES, [])
            features.append(candidates[0]["score"] if candidates else 0)
        elif col == TranslationFeatures.CANDIDATE_GAP:
            candidates = result.get(TranslationFeatures.CANDIDATES, [])
            gap = (candidates[0]["score"] - candidates[1]["score"]) if len(candidates) > 1 else 0
            features.append(gap)
        elif col == TranslationFeatures.IS_EXACT_MATCH:
            features.append(1 if "exact_match" in result.get(TranslationFeatures.REASON, "") else 0)
        elif col == TranslationFeatures.IS_BM25_MATCH:
            bm25 = result.get(TranslationFeatures.BM25_ANALYSIS) or {}
            features.append(1 if bm25.get(TranslationFeatures.BM25_EXACT_RANK) else 0)
        elif col == TranslationFeatures.IS_LLM_FALLBACK:
            features.append(1 if "llm" in result.get(TranslationFeatures.REASON, "").lower() else 0)
        elif col == TranslationFeatures.IS_SEGMENT_EXACT_MATCH:
            # DB에 저장되어 있거나, result에 직접 포함될 수 있음
            features.append(result.get(TranslationFeatures.IS_SEGMENT_EXACT_MATCH, 0))
        elif col == TranslationFeatures.IS_TOP_SEGMENT:
            # top_doc_type이 "cn_segment"이면 1
            candidates = result.get(TranslationFeatures.CANDIDATES, [])
            if candidates:
                top_metadata = candidates[0].get("metadata", {})
                top_doc_type = top_metadata.get("doc_type", "")
                features.append(1 if top_doc_type == "cn_segment" else 0)
            else:
                features.append(0)
        elif col == TranslationFeatures.BM25_EXACT_RANK:
            bm25 = result.get(TranslationFeatures.BM25_ANALYSIS) or {}
            features.append(bm25.get(TranslationFeatures.BM25_EXACT_RANK) or 0)
        elif col == TranslationFeatures.BM25_TOP_RANK_IN_HYBRID:
            bm25 = result.get(TranslationFeatures.BM25_ANALYSIS) or {}
            features.append(bm25.get(TranslationFeatures.BM25_TOP_RANK_IN_HYBRID) or 0)
        elif col == TranslationFeatures.RANK_DIFF:
            bm25 = result.get(TranslationFeatures.BM25_ANALYSIS) or {}
            exact = bm25.get(TranslationFeatures.BM25_EXACT_RANK) or 0
            hybrid = bm25.get(TranslationFeatures.BM25_TOP_RANK_IN_HYBRID) or 0
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
        TranslationFeatures.QUERY: [f"테스트용어{i}" for i in range(n)],
        TranslationFeatures.QUERY_LEN: np.random.randint(2, 15, n),
        TranslationFeatures.TOP_SCORE: np.random.uniform(0, 1, n),
        TranslationFeatures.CANDIDATE_GAP: np.random.uniform(0, 0.3, n),
        TranslationFeatures.IS_EXACT_MATCH: np.random.choice([True, False], n, p=[0.3, 0.7]),
        TranslationFeatures.IS_BM25_MATCH: np.random.choice([True, False], n, p=[0.4, 0.6]),
        TranslationFeatures.IS_LLM_FALLBACK: np.random.choice([True, False], n, p=[0.3, 0.7]),
        TranslationFeatures.IS_SEGMENT_EXACT_MATCH: np.random.choice([True, False], n, p=[0.2, 0.8]),
        TranslationFeatures.IS_TOP_SEGMENT: np.random.choice([True, False], n, p=[0.15, 0.85]),
        TranslationFeatures.BM25_EXACT_RANK: np.random.choice([1, 2, 3, None], n),
        TranslationFeatures.BM25_TOP_RANK_IN_HYBRID: np.random.choice([1, 2, 3, 4, 5, None], n),
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()

