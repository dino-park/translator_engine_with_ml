# core/retriever.py
# Retriever 생성 및 formatting

import warnings
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from core.normalizer import chinese_tokenizer
from utils import get_translator_logger

# BM25 tokenizer deprecation warning 무시 (중국어는 jieba tokenizer 필수)
warnings.filterwarnings("ignore", message=".*tokenizer parameter is deprecated.*")

logger = get_translator_logger("core.retriever")

PRIMARY_LANG_WEIGHT_BOOST = 1.15  # 우선 검색 언어 결과에 가중치 부여

# -----------------------------
# Retriever 생성
# -----------------------------

def build_retrievers(
    cn_index: VectorStoreIndex,
    ko_index: VectorStoreIndex,
    cn_nodes,
    ko_nodes,
    top_k: int = 5
) -> tuple[QueryFusionRetriever, QueryFusionRetriever]:
    """
    CN/KO 별도 컬렉션용 Hybrid Retriever 생성
    
    Args:
        cn_index: CN 컬렉션의 VectorStoreIndex
        ko_index: KO 컬렉션의 VectorStoreIndex
        cn_nodes: CN nodes (BM25용)
        ko_nodes: KO nodes (BM25용)
        top_k: 검색 결과 개수
    
    Returns:
        (cn_retriever, ko_retriever)
    """
    # CN Retriever (중국어 토크나이저 사용)
    cn_vec = cn_index.as_retriever(similarity_top_k=top_k)
    cn_bm25 = BM25Retriever.from_defaults(
        nodes=cn_nodes, 
        similarity_top_k=top_k, 
        tokenizer=chinese_tokenizer
    )
    cn_retriever = QueryFusionRetriever(
        retrievers=[cn_vec, cn_bm25],
        similarity_top_k=top_k,
        num_queries=1,  # LLM 호출 없음
        mode="reciprocal_rerank",
    )
    
    # KO Retriever (한국어는 기본 토크나이저 사용)
    ko_vec = ko_index.as_retriever(similarity_top_k=top_k)
    ko_bm25 = BM25Retriever.from_defaults(
        nodes=ko_nodes, 
        similarity_top_k=top_k,
        # 한국어는 기본 whitespace 토크나이저 사용
    )
    ko_retriever = QueryFusionRetriever(
        retrievers=[ko_vec, ko_bm25],
        similarity_top_k=top_k,
        num_queries=1,
        mode="reciprocal_rerank",
    )
    
    return cn_retriever, ko_retriever


# -----------------------------
# 검색 및 병합
# -----------------------------

def merge_by_entry_id(cn_results: list, ko_results: list, top_k: int = 5, primary_lang: str = None) -> list[dict]:
    """
    CN/KO 검색 결과를 entry_id 기준으로 병합
    
    - 같은 entry_id가 양쪽에서 나오면 score는 max() 사용
    - parent_entry_id가 있으면 (segment) parent 기준으로 그룹핑
    
    Args:
        cn_results: CN 검색 결과 (NodeWithScore 리스트)
        ko_results: KO 검색 결과 (NodeWithScore 리스트)
        top_k: 최종 반환 개수
        primary_lang: 우선 검색 언어("cn" 또는 "ko"), 지정 시 해당 언어 결과에 가중치 부여
    
    Returns:
        [{"entry_id": ..., "cn": ..., "ko": ..., "score": ..., "matched_by": ...}, ...]
    """
    merged = {}  # entry_id -> candidate
    
    # CN과 KO를 개별로 처리하는 Helper 함수
    def process_results(results, lang, apply_weight):
        for node_with_score in results:
            md = node_with_score.node.metadata or {}
            original_score = getattr(node_with_score, "score", 0) or 0
            
            # 가중치 적용
            score = original_score * PRIMARY_LANG_WEIGHT_BOOST if apply_weight else original_score
            
            # entry_id 추출
            entry_id = md.get("entry_id", "")
            parent_entry_id = md.get("parent_entry_id")
            # segment인 경우 parent_entry_id를 key로 사용
            key = parent_entry_id if parent_entry_id else entry_id
            
            if not key:
                continue
            
            # cn/ko 추출 (segment fallback)
            cn = md.get("cn", "") or md.get("parent_cn", "")
            ko = md.get("ko", "") or md.get("parent_ko", "")
            doc_type = md.get("doc_type", "")
            
            # segment 매칭 시 segment 자체의 cn/ko 사용 (parent가 아닌 매칭된 값)
            candidate = {
                "entry_id": key,
                "cn": cn,
                "ko": ko,
                "score": score,
                "matched_by": doc_type,
                "parent_cn": md.get("parent_cn", ""),
                "parent_ko": md.get("parent_ko", ""),
                "source_collection": lang
            }
            
            if apply_weight:
                logger.debug(f"Applied weight to {lang} results: {original_score:.4f} → {score:.4f}")
            
            _merge_candidate(merged, key, candidate)
    
    process_results(cn_results, "cn", apply_weight=(primary_lang == "cn"))
    process_results(ko_results, "ko", apply_weight=(primary_lang == "ko"))
    
    # score 기준 정렬 후 top_k 반환
    sorted_results = sorted(
        merged.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )[:top_k]
    
    return sorted_results
    
    
def _merge_candidate(merged: dict, key:str, candidate: dict):
    """
    병합 로직
    """
    if key not in merged:
        merged[key] = candidate
        return
    
    existing = merged[key]
    doc_type = candidate["matched_by"]
    score = candidate["score"]
    
    # 우선순위: 1) 정확 매칭(segment 아님), 2) score, 3) segment 우선
    is_candidate_segment = "segment" in doc_type
    is_existing_segment = "segment" in existing["matched_by"]
    
    # 기존이 정확 매칭이고 새로운게 segment면 유지 (정확 매칭 우선)
    if not is_existing_segment and is_candidate_segment:
        pass  # 기존 유지
    # 새로운게 정확 매칭이고 기존이 segment면 교체
    elif is_existing_segment and not is_candidate_segment:
        merged[key] = candidate
    # 둘 다 정확 매칭 또는 둘 다 segment면 score로 판단
    elif score > existing["score"]:
        merged[key] = candidate
    elif score == existing["score"]:
        # score가 같으면 segment 우선
        if is_candidate_segment and not is_existing_segment:
            merged[key] = candidate
    else:
        merged[key] = candidate
    

def search(
    query: str,
    cn_retriever: QueryFusionRetriever,
    ko_retriever: QueryFusionRetriever,
    src_lang: str,
    top_k: int = 5,
    fallback_to_two_way: bool = False,
) -> list[dict]:
    """
    src_lang 기반 선택적 검색 후 entry_id 기준 병합(+ fallback 양방향 검색)
    
    Args:
        query: 검색 쿼리
        cn_retriever: CN 컬렉션 retriever
        ko_retriever: KO 컬렉션 retriever
        src_lang: 원본 언어 ("cn", "ko", "unknown")
        top_k: 최종 반환 개수
        fallback_to_two_way: 양방향 검색 사용 여부(True이면 반대 컬렉션도 검색 후 병합)
    
    Returns:
        병합된 검색 결과 리스트
    """
    logger.info("[search] query=%r, src_lang=%s, top_k=%d, fallback_to_two_way=%s", 
                query[:30] if query else "", src_lang, top_k, fallback_to_two_way)
    
    if fallback_to_two_way:
        cn_results = cn_retriever.retrieve(query)
        ko_results = ko_retriever.retrieve(query)
        
        merged = merge_by_entry_id(cn_results, ko_results, top_k, primary_lang=src_lang)
    
    else:
        # src_lang 기반 선택적 검색
        if src_lang == "cn":
            # 중국어 쿼리 → CN 컬렉션만 검색
            cn_results = cn_retriever.retrieve(query)
            ko_results = []
            logger.debug("[search] CN only: %d results", len(cn_results))
        elif src_lang == "ko":
            # 한국어 쿼리 → KO 컬렉션만 검색
            cn_results = []
            ko_results = ko_retriever.retrieve(query)
            logger.debug("[search] KO only: %d results", len(ko_results))
        else:
            # unknown → 양방향 검색 (fallback)
            cn_results = cn_retriever.retrieve(query)
            ko_results = ko_retriever.retrieve(query)
            logger.debug("[search] Both collections: CN=%d, KO=%d", 
                        len(cn_results), len(ko_results))
        
        # entry_id 기준 병합 (중복 제거 및 정렬)
        merged = merge_by_entry_id(cn_results, ko_results, top_k, primary_lang=None)
    
    logger.info("[search] merged results: %d", len(merged))
    return merged


# -----------------------------
# 결과 포맷팅
# -----------------------------

def format_candidates(merged_results: list[dict], show: int = 5) -> list[dict]:
    """
    병합된 검색 결과를 최종 포맷으로 변환
    
    Returns:
        [{"score": ..., "cn": ..., "ko": ..., "matched_by": ..., "metadata": {...}}, ...]
    """
    formatted = []
    for item in merged_results[:show]:
        formatted.append({
            "score": item["score"],
            "cn": item["cn"],
            "ko": item["ko"],
            "matched_by": item["matched_by"],
            "metadata": {
                "entry_id": item.get("entry_id"),
                "parent_cn": item.get("parent_cn"),
                "parent_ko": item.get("parent_ko"),
                "doc_type": item.get("matched_by"),  # doc_type 추가 (matched_by와 동일)
            }
        })
    return formatted
