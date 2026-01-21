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