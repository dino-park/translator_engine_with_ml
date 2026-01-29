import os

from dotenv import load_dotenv
from core.embedding_model import CompassEmbeddingModel
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings


# RRF(Reciprocal Rank Fusion) score threshold
# RRF score = Σ(1/(rank+k)), k=60 기준으로 상위 2~3위 이내 매칭을 신뢰
# - 1위 (2개 검색기): ≈0.033
# - 5위 이하: ≈0.031 미만
SCORE_THRESHOLD = 0.025


# ===============================================
# Translation Reason 상수
# - 번역 결과의 reason 필드에 사용되는 값들
# - 중앙 관리를 통해 일관성 유지 및 오타 방지
# ===============================================
class TranslationReason:
    """번역 결과의 reason 값 정의 (단순 상수 모음)"""
    
    # === Term 번역 (translate_term) ===
    EXACT_MATCH = "exact_match"                          # 정확 매칭 성공
    LLM_TERM_FALLBACK_NO_MATCH = "llm_term_fallback_no_match"  # 결과 없음 → LLM 번역
    LLM_TERM_FALLBACK_NO_EXACT_MATCH = "llm_term_fallback_no_exact_match"  # 정확 매칭 없음 → LLM 번역
    NO_MATCH = "no_match"                                # 결과 없음, LLM도 실패
    NO_KO_AND_LLM_FAILED = "no_ko_and_llm_failed"        # ko 없음, LLM도 실패
    TOP_CANDIDATE_HIGH_CONFIDENCE = "top_candidate_high_confidence"  # 1위 신뢰
    
    # === Sentence 번역 (translate_sentence_with_glossary) ===
    TM_EXACT_REUSE = "tm_exact_reuse"                    # TM 정확 매칭 재사용
    TM_PARTIAL_REUSE = "tm_partial_reuse"                # TM 유사 문장 참고
    GLOSSARY_AWARE_TRANSLATION = "glossary_aware_translation"  # 용어사전 참고 번역
    LLM_TRANSLATION = "llm_translation"                  # 순수 LLM 번역
    LLM_NOT_AVAILABLE = "llm_not_available"              # LLM 사용 불가
    LLM_FAILED = "llm_failed"                            # LLM 번역 실패
    
    # === 기타 ===
    NO_TRANSLATION = "no_translation"                    # 번역 대상 없음
    UNKNOWN_MODE = "unknown_mode"                        # 알 수 없는 모드


# ===============================================
# Translation Features 상수
# - 번역 로그 DB 및 ML 학습에 사용되는 feature 이름들
# - 중앙 관리를 통해 일관성 유지 및 오타 방지
# - 사용처: translation_logger_db.py, weak_supervision_labeler.py, train_confidence_model.py 등
# ===============================================
class TranslationFeatures:
    """
    번역 결과 분석 및 ML 학습에 사용되는 feature 키 정의
    
    Features는 TranslationReason과 달리 하나의 번역에 여러 개가 동시에 기록됩니다.
    - TranslationReason: "어떤 방식으로 번역했는가?" (1개, categorical)
    - TranslationFeatures: "번역 과정에서 무슨 일이 일어났는가?" (여러 개, numeric/boolean)
    
    사용 예시:
        # 기존 방식 (여전히 사용 가능)
        is_exact = row.get("is_exact_match")
        
        # 새로운 방식 (타입 안전, 자동완성, 오타 방지)
        is_exact = row.get(TranslationFeatures.IS_EXACT_MATCH)
    """
    
    # === 매칭 유형 (Boolean Features) ===
    IS_EXACT_MATCH = "is_exact_match"           # 정확 매칭 성공 (normalized 기준)
    IS_BM25_MATCH = "is_bm25_match"             # BM25에서 정확 매칭 발견
    IS_LLM_FALLBACK = "is_llm_fallback"         # 용어사전 미발견 → LLM 번역 사용
    
    # === BM25 분석 (Rank Features) ===
    # BM25와 Vector의 랭킹 차이를 분석하여 검색 품질 평가
    BM25_EXACT_RANK = "bm25_exact_rank"                     # BM25 단독 검색에서 정확 매칭의 순위 (None이면 없음)
    BM25_TOP_RANK_IN_HYBRID = "bm25_top_rank_in_hybrid"     # BM25 1등이 Hybrid 검색에서 몇 등인지 (Vector 영향 확인)
    
    # === 검색 점수 (Score Features) ===
    TOP_SCORE = "top_score"                     # 1위 후보의 검색 점수 (높을수록 신뢰)
    CANDIDATE_GAP = "candidate_gap"             # 1위-2위 점수 차이 (클수록 확실한 1위)
    BM25_TOP_SCORE = "bm25_top_score"           # BM25 검색 1위 점수
    
    # === 입력 정보 ===
    QUERY = "query"                             # 사용자 입력 쿼리
    QUERY_LEN = "query_len"                     # 쿼리 길이 (짧으면 용어, 길면 문장)
    SRC_LANG = "src_lang"                       # 원본 언어 (cn/ko)
    MODE = "mode"                               # term(용어) 또는 sentence(문장)
    
    # === 출력 정보 ===
    TRANSLATION = "translation"          
    # 번역 결과
    REASON = "reason"                           # 번역 방식 (TranslationReason 참조)
    
    # === Document Type Features ===
    TOP_DOC_TYPE = "top_doc_type"               # 1위 후보의 문서 타입 (cn/cn_segment/cn_normalized/ko 등)
    IS_SEGMENT_EXACT_MATCH = "is_segment_exact_match"  # cn_segment인 경우 정확 매칭 여부
    SEGMENT_PARENT_CN = "segment_parent_cn"     # cn_segment의 원문 (parent_cn)
    IS_TOP_SEGMENT = "is_top_segment"           # 1위가 segment 매칭인지 (weak_supervision용)
    
    # === Glossary Hints Features ===
    HAS_GLOSSARY_HINTS = "has_glossary_hints"   # 용어사전 힌트 존재 여부
    GLOSSARY_MATCH_COUNT = "glossary_match_count"  # 용어사전 매칭 개수
    
    # === 신뢰도 체크 Features ===
    PASSED_BM25_CHECK = "passed_bm25_check"     # BM25 신뢰도 체크 통과 여부
    PASSED_GAP_CHECK = "passed_gap_check"       # Candidate gap 체크 통과 여부
    
    # === Cluster Analysis Features (weak_supervision_labeler.py) ===
    RESULT_COUNT = "result_count"               # 검색 결과 개수
    RANK_DIFF = "rank_diff"                     # BM25_EXACT_RANK - BM25_TOP_RANK_IN_HYBRID (양수면 Vector가 다른 것 선호)
    
    WEAK_LABEL = "weak_label"                   # Weak Label(1: GOOD, 0: BAD, -1: ABSTAIN)
    LABEL_STRENGTH = "label_strength"           # Weak Label 강도 (GOOD - BAD)
    IS_CONFLICT = "is_conflict"                 # 규칙 간 충돌 여부
    LABEL_SOURCE = "label_source"               # 최종 레이블을 결정한 규칙 이름
    
    # === Metadata ===
    CANDIDATES = "candidates"                   # 검색 후보 리스트 (JSON)
    CANDIDATES_JSON = "candidates_json"         # 검색 후보 리스트 (JSON string for DB)
    BM25_ANALYSIS = "bm25_analysis"             # BM25 분석 결과 (dict)
    GLOSSARY_HINTS = "glossary_hints"           # 용어사전 힌트 정보
    TM_MATCH = "tm_match"                       # Translation Memory 매칭 정보
    
    # === Performance ===
    RESPONSE_TIME_MS = "response_time_ms"       # 응답 시간 (밀리초)
    
    # === Source ===
    SOURCE = "source"                           # 요청 출처 (api/debug/batch)
    CREATED_AT = "created_at"                   # 생성 시간


def load_env():
    """
    .env 파일에서 환경변수 로드

    """
    load_dotenv()
    env = {
        "COMPASS_API_KEY": os.getenv("COMPASS_API_KEY"),
        "COMPASS_BASE_URL": os.getenv("COMPASS_BASE_URL"),
        "LLM_MODEL": os.getenv("LLM_MODEL"),
        "PERSIST_DIR": os.getenv("PERSIST_DIR"),        
    }
    if not env["COMPASS_API_KEY"]:
        raise RuntimeError("COMPASS_API_KEY is not set")
    return env


def setup_settings(env: dict):
    """
    LlamaIndex 전역 Settings 설정
    - embedding model: compass-embedding-v4
    - dimensions: 2560
    - chunk size: 300
    - chunk overlap: 30
    - llm: 사용자 지정 LLM
    """
    Settings.env = env  # env를 Settings에 저장
    
    # embed_batch_size: Rate Limit 문제로 20으로 줄임 (기존 100 → 64 → 50 → 20)
    # 재시도 로직과 exponential backoff가 _embed_with_text_type에 구현됨
    Settings.embed_model = CompassEmbeddingModel(
        api_key=env["COMPASS_API_KEY"],
        api_base=env["COMPASS_BASE_URL"],
        model="compass-embedding-v4",
        dimensions=2560,
        embed_batch_size=100
    )
    
    Settings.chunk_size = 300
    Settings.chunk_overlap = 30
    

    Settings.llm = LlamaOpenAI(
        api_key=env["COMPASS_API_KEY"],
        api_base=env["COMPASS_BASE_URL"],
        model=env["LLM_MODEL"],
        temperature=0.0
    )
    return Settings