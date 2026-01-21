import logging
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from core.init_document_node import (
    load_dataset,
    build_documents,
    build_nodes,
    save_nodes_cache,
)
from core.chroma_utils import add_user_entry_to_glossary
from core.embedding_model import CompassEmbeddingModel


logger = logging.getLogger("translator")

BASE_CSV_PATH = "glossary/Fellowmoon 语言包1211_기본단어.csv"


# -----------------------------
# Chroma 초기화
# -----------------------------

def _init_single_collection(
    persist_dir: str, 
    collection_name: str, 
    client: chromadb.PersistentClient = None
) -> ChromaVectorStore:
    """
    단일 Chroma 컬렉션 초기화 (내부용)
    """
    if client is None:
        client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    return ChromaVectorStore(chroma_collection=collection)


def init_chroma(persist_dir: str, clear: bool = False) -> tuple[ChromaVectorStore, ChromaVectorStore]:
    """
    CN/KO 별도 컬렉션 초기화
    
    Args:
        persist_dir: 저장 디렉토리
        clear: True면 기존 컬렉션 삭제 후 새로 생성
    
    Returns:
        (cn_vector_store, ko_vector_store)
    """
    # 단일 클라이언트 생성 (캐싱 문제 방지)
    client = chromadb.PersistentClient(path=persist_dir)
    
    if clear:
        # 동일한 클라이언트에서 삭제 후 재생성
        for name in ["game-translation-cn", "game-translation-ko", "game-translation"]:
            try:
                client.delete_collection(name=name)
                logger.info("[clear] Deleted collection '%s'", name)
            except Exception:
                logger.info("[clear] No collection '%s' to delete", name)
    
    cn_store = _init_single_collection(persist_dir, "game-translation-cn", client)
    ko_store = _init_single_collection(persist_dir, "game-translation-ko", client)
    return cn_store, ko_store


# -----------------------------
# index 생성/로딩
# -----------------------------

def build_index_from_nodes(nodes, vector_store: ChromaVectorStore) -> VectorStoreIndex:
    """
    Nodes를 사용하여 VectorStoreIndex 생성(최초 인덱싱 시 사용)
    - CompassEmbeddingModel로 API 응답 기반 토큰 수 추적
    """
    # 토큰 카운트 리셋 (이전 인덱싱에서 누적된 값 초기화)
    embed_model = Settings.embed_model
    if isinstance(embed_model, CompassEmbeddingModel):
        embed_model.reset_token_count()
    
    storage = StorageContext.from_defaults(vector_store=vector_store)
    # 이 단계에서 임베딩 모델을 사용하여 노드를 임베딩
    index = VectorStoreIndex(
        nodes=nodes, 
        storage_context=storage
    )
    
    # API 응답 기반 정확한 토큰 수 로깅
    if isinstance(embed_model, CompassEmbeddingModel):
        logger.info(
            "Index created: nodes=%d, embedding_tokens=%d (from API)",
            len(nodes),
            embed_model.total_tokens
        )
    else:
        logger.info("Index created: nodes=%d", len(nodes))
    
    return index


def run_indexing(csv_path: str = BASE_CSV_PATH, clear: bool = False):
    """
    CN/KO 별도 컬렉션으로 인덱싱
    
    1. CSV에서 데이터 로드
    2. CN/KO Documents 분리 생성
    3. CN/KO 각각 Node 생성
    4. CN/KO 별도 컬렉션에 인덱싱
    5. CN/KO 별도 캐시 저장
    
    Args:
        csv_path: CSV 파일 경로
        clear: True면 기존 컬렉션 삭제 후 새로 인덱싱
    """
    logger.info("[indexing] start. csv=%s, clear=%s", csv_path, clear)
    
    persist_dir = Settings.env["PERSIST_DIR"]
    
    # 데이터 로드 및 Documents 생성
    df = load_dataset(csv_path)
    cn_docs, ko_docs = build_documents(df)
    
    # Nodes 생성
    cn_nodes = build_nodes(cn_docs)
    ko_nodes = build_nodes(ko_docs)
    
    # CN/KO 별도 컬렉션 초기화 (clear 옵션 전달)
    cn_store, ko_store = init_chroma(persist_dir, clear=clear)
    
    # 인덱싱
    logger.info("[indexing] Indexing CN collection...")
    _ = build_index_from_nodes(cn_nodes, cn_store)
    
    logger.info("[indexing] Indexing KO collection...")
    _ = build_index_from_nodes(ko_nodes, ko_store)
    
    # 캐시 저장 (BM25용)
    save_nodes_cache(cn_nodes, ko_nodes)
    
    logger.info(
        "[indexing] finished. rows=%d, cn_docs=%d, ko_docs=%d, cn_nodes=%d, ko_nodes=%d",
        len(df), len(cn_docs), len(ko_docs), len(cn_nodes), len(ko_nodes)
    )


def run_indexing_for_text(cn: str = "", ko: str = ""):
    """
    단순 텍스트 문자열을 인덱싱 (API/웹앱에서 동적으로 용어 추가)
    
    Args:
        cn (str): 중국어 텍스트 (둘 중 하나는 필수)
        ko (str): 한국어 텍스트 (둘 중 하나는 필수)
    
    사용 예시:
        # 양방향
        run_indexing_for_text("超现象管理局", "초현상 관리국")
        
        # 중국어만
        run_indexing_for_text(cn="简易突破模块组")
        
        # 한국어만
        run_indexing_for_text(ko="월상석 가루")
    """
    cn = cn.strip() if cn else ""
    ko = ko.strip() if ko else ""
    
    if not cn and not ko:
        logger.warning("[index_text] cn 또는 ko 중 하나는 필수입니다")
        return
    
    # add_user_entry_to_glossary 활용
    add_user_entry_to_glossary(
        cn=cn,
        ko=ko,
        doc_type="text_input",
        src_lang="cn" if cn else "ko",
        reason="manual_index_text"
    )
    
    logger.info("[index_text] Added: cn=%r, ko=%r", cn, ko)
