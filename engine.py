import time
import warnings

from core.config import Settings
from core.config import load_env, setup_settings, TranslationReason
from core.init_document_node import load_nodes_cache
from core.retriever import build_retrievers
from core.glossary import build_glossary_map
from core.translator import translate_sentence_with_glossary, translate_term
from llama_index.core import Settings, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from preprocessor import (
    detect_intent, preprocess_query_smart
)
from utils import get_translator_logger
from indexing import init_chroma

# BM25 tokenizer deprecation warning 무시
warnings.filterwarnings("ignore", message=".*tokenizer parameter is deprecated.*")

logger = get_translator_logger("engine")


# 엔진 캐시 (Lazy 초기화용)
_engine_cache = {
    "cn_index": None,
    "ko_index": None,
    "cn_nodes": None,
    "ko_nodes": None,
    "cn_retriever": None,
    "ko_retriever": None,
    "cn_bm25_retriever": None,  # CN BM25 전용 retriever (translate_term용)
    "ko_bm25_retriever": None,  # KO BM25 전용 retriever (translate_term용)
    "glossary_map": None,
    "initialized": False,
}


def init_engine():
    """
    번역 엔진 초기화 - CN/KO 별도 컬렉션 사용
    
    Returns:
        (cn_retriever, ko_retriever) 튜플
    """
    env = load_env()
    setup_settings(env)
    
    persist_dir = Settings.env["PERSIST_DIR"]
    
    # CN/KO 별도 컬렉션 로드
    cn_store, ko_store = init_chroma(persist_dir)
    cn_index = VectorStoreIndex.from_vector_store(cn_store)
    ko_index = VectorStoreIndex.from_vector_store(ko_store)
    
    # CN/KO nodes 캐시 로드
    cn_nodes, ko_nodes = load_nodes_cache()
    if cn_nodes is None or ko_nodes is None:
        raise RuntimeError(
            "Nodes cache not found. Run run_indexing() first."
        )
    
    # Retrievers 생성
    cn_retriever, ko_retriever = build_retrievers(
        cn_index, ko_index, cn_nodes, ko_nodes
    )
    
    # BM25 전용 retriever 생성 (translate_term의 로깅/분석용)
    from core.normalizer import chinese_tokenizer
    cn_bm25_retriever = BM25Retriever.from_defaults(
        nodes=cn_nodes,
        similarity_top_k=5,
        tokenizer=chinese_tokenizer
    )
    ko_bm25_retriever = BM25Retriever.from_defaults(
        nodes=ko_nodes,
        similarity_top_k=5,
        # 한국어는 기본 whitespace 토크나이저 사용
    )
    
    # 캐시에 저장
    _engine_cache["cn_index"] = cn_index
    _engine_cache["ko_index"] = ko_index
    _engine_cache["cn_nodes"] = cn_nodes
    _engine_cache["ko_nodes"] = ko_nodes
    _engine_cache["cn_retriever"] = cn_retriever
    _engine_cache["ko_retriever"] = ko_retriever
    _engine_cache["cn_bm25_retriever"] = cn_bm25_retriever
    _engine_cache["ko_bm25_retriever"] = ko_bm25_retriever
    _engine_cache["initialized"] = True
    
    # glossary map도 생성 (부분 매칭용)
    glossary_map = build_glossary_map()
    _engine_cache["glossary_map"] = glossary_map
    
    logger.info(
        "Engine initialized: cn_nodes=%d, ko_nodes=%d",
        len(cn_nodes), len(ko_nodes)
    )
    return cn_retriever, ko_retriever


def translate_execute(raw_query: str) -> dict:
    """
    번역 실행
    
    Args:
        raw_query: 원본 쿼리
    
    Returns:
        번역 결과 딕셔너리
    """
    start_time = time.time()
    
    if not _engine_cache["initialized"]:
        raise RuntimeError("Engine not initialized. Call init_engine() first.")
    
    # Cache 로드하여 retriever 가져오기
    cn_retriever = _engine_cache["cn_retriever"]
    ko_retriever = _engine_cache["ko_retriever"]
    
    intent = detect_intent(raw_query)
    mode = intent["mode"]
    src_lang = intent["src_lang"]
    
    if mode == "term":
        # 단어 모드: 단방향 -> 조건부 양방향
        preprocessed_query = preprocess_query_smart(raw_query, llm=Settings.llm)
        
        # translate_term() 호출 (양방향 검색, 모든 feature 포함)
        cn_bm25_retriever = _engine_cache["cn_bm25_retriever"]
        ko_bm25_retriever = _engine_cache["ko_bm25_retriever"]
        glossary_map = _engine_cache["glossary_map"]
        
        result = translate_term(
            query=preprocessed_query,
            cn_retriever=cn_retriever,
            ko_retriever=ko_retriever,
            src_lang=src_lang,
            use_llm_fallback=True,
            cn_bm25_retriever=cn_bm25_retriever,
            ko_bm25_retriever=ko_bm25_retriever,
            glossary_map=glossary_map
        )
        
        # 메타 정보 추가
        result["raw_query"] = raw_query
        result["mode"] = mode
        result["response_time_ms"] = int((time.time() - start_time) * 1000)
        
        logger.info(
            "query=%r, translation=%r, reason=%r",
            preprocessed_query, result.get('translation'), result.get('reason')
        )
        return result
    
    elif mode == "sentence":
        # 문장 모드 (glossary 기반 번역): 항상 양방향 검색
        result = translate_sentence_with_glossary(
            raw_query,
            cn_retriever,
            ko_retriever,
            src_lang=src_lang, 
            use_llm=True,
            tm_retriever=None,
        )
        result["raw_query"] = raw_query
        result["mode"] = mode
        result["response_time_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    return {
        "raw_query": raw_query,
        "mode": mode,
        "src_lang": src_lang,
        "translation": None,
        "reason": TranslationReason.UNKNOWN_MODE,
        "response_time_ms": int((time.time() - start_time) * 1000),
    }
