# core/chroma_utils.py
# Chroma DB 관리 유틸리티

import chromadb

from llama_index.core import Settings
from utils import get_translator_logger

logger = get_translator_logger("core.chroma_utils")

# ===============================================
# 런타임 user_term 메모리 캐시 (suffix 검색용)
# - ChromaDB 저장과 동시에 메모리에도 캐싱
# - suffix 검색 시 실시간 반영 (API 호출 없음)
# ===============================================
_user_term_cache: list[dict] = []

def get_user_term_cache() -> list[dict]:
    """user_term 메모리 캐시 반환 (suffix 검색용)"""
    return _user_term_cache

def _add_to_user_term_cache(cn: str, ko: str, doc_type: str):
    """user_term을 메모리 캐시에 추가"""
    _user_term_cache.append({
        "cn": cn,
        "ko": ko,
        "doc_type": doc_type
    })
    logger.debug("Added to user_term cache: cn=%r, ko=%r (total=%d)", cn, ko, len(_user_term_cache))

# ===============================================
# 사용자 입력 → Glossary 자동 확장 (학습 기능)
# ===============================================

def get_chroma_collection(persist_dir: str) -> chromadb.Collection:
    """Chroma 컬렉션 직접 접근 (저장용)"""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name="game-translation")


def add_user_entry_to_glossary(
    cn: str,
    ko: str,
    doc_type: str,  # "user_term" 또는 "user_sentence"
    src_lang: str,
    reason: str = ""
) -> str | None:
    """
    사용자 입력 + 번역 결과를 기존 glossary에 추가
    
    Args:
        cn: 중국어 텍스트
        ko: 한국어 텍스트
        doc_type: 문서 유형 ("user_term", "user_sentence")
        src_lang: 원본 언어 ("cn" 또는 "ko")
        reason: 번역 방식 (llm_fallback 등)
    
    Returns:
        저장된 document ID (실패 시 None)
    """
    try:
        from datetime import datetime
        
        # 1. 고유 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"{doc_type}-{timestamp}-{hash(cn + ko) % 10000}"
        
        # 2. 검색 기준 텍스트 결정 (원본 언어 기준)
        search_text = cn if src_lang == "cn" else ko
        
        # 3. 임베딩 생성
        embedding = Settings.embed_model.get_text_embedding(search_text)
        
        # 4. 메타데이터 구성
        metadata = {
            "cn": cn,
            "ko": ko,
            "doc_type": doc_type,
            "src_lang": src_lang,
            "reason": reason,
            "created_at": timestamp
        }
        
        # 5. Chroma 컬렉션에 직접 추가
        collection = get_chroma_collection(Settings.env["PERSIST_DIR"])
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[search_text],
            metadatas=[metadata]
        )
        
        # 6. 메모리 캐시에도 추가 (suffix 검색용, 실시간 반영)
        _add_to_user_term_cache(cn, ko, doc_type)
        
        logger.info(
            "Added to glossary: id=%s, doc_type=%s, cn=%r, ko=%r",
            doc_id, doc_type, cn[:20] + "..." if len(cn) > 20 else cn, 
            ko[:20] + "..." if len(ko) > 20 else ko
        )
        return doc_id
    
    except Exception as e:
        logger.error("Failed to add user entry to glossary: %s", e)
        return None


def delete_from_glossary_by_metadata(
    cn: str = None,
    ko: str = None,
    doc_type: str = None
) -> int:
    """
    메타데이터 조건에 맞는 항목을 VectorDB에서 삭제
    
    Args:
        cn: 삭제할 중국어 텍스트 (정확 매칭)
        ko: 삭제할 한국어 텍스트 (정확 매칭)
        doc_type: 문서 유형 필터 (예: "user_term", "user_sentence")
    
    Returns:
        삭제된 항목 수
    
    Example:
        # 특정 잘못된 번역 삭제
        delete_from_glossary_by_metadata(cn="综合速配", ko="종합 매칭")
        
        # 모든 user_term 삭제
        delete_from_glossary_by_metadata(doc_type="user_term")
    """
    try:
        collection = get_chroma_collection(Settings.env["PERSIST_DIR"])
        
        # where 조건 구성
        where_conditions = []
        if cn:
            where_conditions.append({"cn": cn})
        if ko:
            where_conditions.append({"ko": ko})
        if doc_type:
            where_conditions.append({"doc_type": doc_type})
        
        if not where_conditions:
            logger.warning("No filter conditions provided for deletion")
            return 0
        
        # 여러 조건이 있으면 $and로 결합
        if len(where_conditions) == 1:
            where = where_conditions[0]
        else:
            where = {"$and": where_conditions}
        
        # 먼저 삭제할 항목 조회
        results = collection.get(where=where)
        count = len(results["ids"]) if results["ids"] else 0
        
        if count == 0:
            logger.info("No entries found matching: cn=%r, ko=%r, doc_type=%r", cn, ko, doc_type)
            return 0
        
        # 삭제 실행
        collection.delete(where=where)
        
        logger.info(
            "Deleted %d entries from glossary: cn=%r, ko=%r, doc_type=%r",
            count, cn, ko, doc_type
        )
        return count
    
    except Exception as e:
        logger.error("Failed to delete from glossary: %s", e)
        return 0


def list_user_entries(doc_type: str = None, limit: int = 50) -> list[dict]:
    """
    사용자가 추가한 glossary 항목 목록 조회
    
    Args:
        doc_type: 필터링할 문서 유형 ("user_term" 또는 "user_sentence")
        limit: 최대 반환 개수
    
    Returns:
        [{"id": ..., "cn": ..., "ko": ..., "doc_type": ..., "created_at": ...}, ...]
    """
    try:
        collection = get_chroma_collection(Settings.env["PERSIST_DIR"])
        
        # user_로 시작하는 doc_type만 조회
        if doc_type:
            where = {"doc_type": doc_type}
        else:
            where = {"$or": [{"doc_type": "user_term"}, {"doc_type": "user_sentence"}]}
        
        results = collection.get(where=where, limit=limit)
        
        entries = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            entries.append({
                "id": doc_id,
                "cn": meta.get("cn", ""),
                "ko": meta.get("ko", ""),
                "doc_type": meta.get("doc_type", ""),
                "created_at": meta.get("created_at", ""),
            })
        
        return entries
    
    except Exception as e:
        logger.error("Failed to list user entries: %s", e)
        return []
    
    