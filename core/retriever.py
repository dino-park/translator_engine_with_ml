# core/retriever.py
# Retriever 생성 및 formatting

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from core.normalizer import chinese_tokenizer
from utils import get_translator_logger


logger = get_translator_logger("core.retriever")


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

def merge_by_entry_id(cn_results: list, ko_results: list, top_k: int = 5) -> list[dict]:
    """
    CN/KO 검색 결과를 entry_id 기준으로 병합
    
    - 같은 entry_id가 양쪽에서 나오면 score는 max() 사용
    - parent_entry_id가 있으면 (segment) parent 기준으로 그룹핑
    
    Args:
        cn_results: CN 검색 결과 (NodeWithScore 리스트)
        ko_results: KO 검색 결과 (NodeWithScore 리스트)
        top_k: 최종 반환 개수
    
    Returns:
        [{"entry_id": ..., "cn": ..., "ko": ..., "score": ..., "matched_by": ...}, ...]
    """
    merged = {}  # entry_id -> candidate
    
    all_results = list(cn_results) + list(ko_results)
    
    for node_with_score in all_results:
        md = node_with_score.node.metadata or {}
        score = getattr(node_with_score, "score", 0) or 0
        
        entry_id = md.get("entry_id", "")
        parent_entry_id = md.get("parent_entry_id")
        
        # segment인 경우 parent_entry_id를 key로 사용
        key = parent_entry_id if parent_entry_id else entry_id
        
        if not key:
            continue
        
        cn = md.get("cn", "")
        ko = md.get("ko", "")
        parent_cn = md.get("parent_cn")
        parent_ko = md.get("parent_ko")
        doc_type = md.get("doc_type", "")
        
        # segment 매칭 시 segment 자체의 cn/ko 사용 (parent가 아닌 매칭된 값)
        candidate = {
            "entry_id": key,
            "cn": cn,
            "ko": ko,
            "score": score,
            "matched_by": doc_type,
            "parent_cn": parent_cn,
            "parent_ko": parent_ko,
        }
        
        if key in merged:
            existing = merged[key]
            # score가 더 높으면 교체, 같으면 segment 우선
            if score > existing["score"]:
                merged[key] = candidate
            elif score == existing["score"]:
                # segment 매칭을 우선
                if "segment" in doc_type and "segment" not in existing["matched_by"]:
                    merged[key] = candidate
        else:
            merged[key] = candidate
    
    # score 기준 정렬 후 top_k 반환
    sorted_results = sorted(
        merged.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )[:top_k]
    
    return sorted_results


def search(
    query: str,
    cn_retriever: QueryFusionRetriever,
    ko_retriever: QueryFusionRetriever,
    top_k: int = 5
) -> list[dict]:
    """
    CN/KO 양쪽에서 검색 후 entry_id 기준 병합
    
    Args:
        query: 검색 쿼리
        cn_retriever: CN 컬렉션 retriever
        ko_retriever: KO 컬렉션 retriever
        top_k: 최종 반환 개수
    
    Returns:
        병합된 검색 결과 리스트
    """
    logger.info("[search] query=%r, top_k=%d", query[:30] if query else "", top_k)
    
    # CN 검색
    cn_results = cn_retriever.retrieve(query)
    logger.debug("[search] CN results: %d", len(cn_results))
    
    # KO 검색
    ko_results = ko_retriever.retrieve(query)
    logger.debug("[search] KO results: %d", len(ko_results))
    
    # entry_id 기준 병합
    merged = merge_by_entry_id(cn_results, ko_results, top_k)
    
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
