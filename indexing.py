# indexing.py
# 인덱싱 관련 함수 정의
import logging
import chromadb
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from core.init_document_node import (
    load_dataset,
    build_documents,
    build_nodes,
    save_nodes_cache,
    save_nodes_cache_incremental
)
from core.chroma_utils import add_user_entry_to_glossary
from core.embedding_model import CompassEmbeddingModel


logger = logging.getLogger("translator")

# Glossary 디렉토리 경로
GLOSSARY_DIR = "glossary"


def load_all_csvs_from_directory(directory: str = GLOSSARY_DIR) -> pd.DataFrame:
    """
    지정된 디렉토리의 모든 CSV 파일을 로드하여 병합
    - 중복 제거 (cn 기준)
    - 순서 유지
    
    Args:
        directory: CSV 파일이 있는 디렉토리 경로
    
    Returns:
        병합된 DataFrame
    
    Raises:
        ValueError: CSV 파일이 없거나 로드 실패 시
    """
    csv_files = sorted(glob.glob(f"{directory}/*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")
    
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")
    
    all_dfs = []
    for csv_path in csv_files:
        try:
            df = load_dataset(csv_path)
            all_dfs.append(df)
            logger.info(f"  ✓ {Path(csv_path).name}: {len(df)} rows")
            print(f"  ✓ {Path(csv_path).name}: {len(df):,}개 행")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {Path(csv_path).name}: {e}")
            print(f"  ✗ {Path(csv_path).name}: 로드 실패 ({e})")
            # 에러 발생해도 다른 파일은 계속 처리
    
    if not all_dfs:
        raise ValueError(f"No valid CSV files loaded from {directory}")
    
    # 병합
    merged = pd.concat(all_dfs, ignore_index=True)
    before_count = len(merged)
    
    # 중복 제거 (cn 기준, 첫 번째 항목 유지)
    merged = merged.drop_duplicates(subset="cn", keep="first")
    after_count = len(merged)
    
    duplicates_removed = before_count - after_count
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate entries")
        print(f"  → 중복 제거: {duplicates_removed:,}개")
    
    logger.info(f"Total merged rows: {after_count}")
    
    return merged

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
    
    # === 진행도 표시 ===
    # ================================================
    print(f"\n🔄 임베딩 시작: 총 {len(nodes):,}개 노드")
    print(f"   (배치 크기: {getattr(embed_model, 'embed_batch_size', 'N/A')}개씩 처리)")
    
    # 임베딩 진행도 표시
    with tqdm(total=len(nodes), desc="📝 임베딩", unit="nodes", ncols=80) as pbar:
        # VectorStoreIndex 생성 시 내부적으로 embed_batch_size 단위로 임베딩 진행
        index = VectorStoreIndex(
            nodes=nodes, 
            storage_context=storage,
            show_progress=False  # LlamaIndex 내부 진행도는 끄고 수동 제어
        )
        pbar.update(len(nodes))  # 완료 시 전체 업데이트
    # ================================================
    
    # API 응답 기반 정확한 토큰 수 로깅
    if isinstance(embed_model, CompassEmbeddingModel):
        logger.info(
            "Index created: nodes=%d, embedding_tokens=%d (from API)",
            len(nodes),
            embed_model.total_tokens
        )
        print(f"✅ 임베딩 완료: {len(nodes):,}개 노드, {embed_model.total_tokens:,} 토큰")        
    else:
        logger.info("Index created: nodes=%d", len(nodes))
        print(f"✅ 임베딩 완료: {len(nodes):,}개 노드")        
    
    return index


def run_indexing(csv_path: str = None, clear: bool = False):
    """
    CN/KO 별도 컬렉션으로 인덱싱
    
    1. CSV에서 데이터 로드
    2. CN/KO Documents 분리 생성
    3. CN/KO 각각 Node 생성
    4. CN/KO 별도 컬렉션에 인덱싱
    5. CN/KO 별도 캐시 저장
    
    Args:
        csv_path: CSV 파일 경로 (None이면 glossary 폴더의 모든 CSV 파일)
        clear: True면 기존 컬렉션 삭제 후 새로 인덱싱
    
    사용예시:
        # 모든 CSV 파일 인덱싱(전체 ReIndexing)
        run_indexing(clear=True)
        # 특정 파일만 증분 인덱싱
        run_indexing(csv_path="glossary/new_terms.csv", clear=False)
    """
    print("\n" + "="*60)
    print("📚 Glossary 인덱싱 시작")
    print("="*60)
    
    logger.info("[indexing] start. csv=%s, clear=%s", csv_path, clear)
    
    persist_dir = Settings.env["PERSIST_DIR"]
    
    # 1. CSV파일 결정 후 데이터 로드
    if csv_path is None:
        # 모든 CSV파일 자동 로드(Default action)
        logger.info("[indexing] Loading all CSV files from %s", GLOSSARY_DIR)
        csv_files = sorted(glob.glob(f"{GLOSSARY_DIR}/*.csv"))
        print(f"\n[1/7] CSV 파일 검색: {len(csv_files)}개 파일 발견")
        for f in csv_files:
            print(f"  - {Path(f).name}")
        
        print("\n[2/7] CSV 로드 및 병합 중...")
        df = load_all_csvs_from_directory(GLOSSARY_DIR)
        print(f"✅ {len(df):,}개 행 로드 완료(중복 제거 완료)")
        
    else:
        # 단일 파일 로드
        logger.info("[indexing] Loading single CSV file: %s", csv_path)
        print(f"\n[1/7] CSV 로드 중... ({Path(csv_path).name})")
        df = load_dataset(csv_path)
        print(f"✅ {len(df):,}개 행 로드 완료")
        print("\n[2/7] 단일 파일 모드 (병합 단계 건너뜀)")

    # 2. Documents 생성
    print("\n[3/7] Documents 생성 중...")
    cn_docs, ko_docs = build_documents(df)
    print(f"✅ CN: {len(cn_docs):,}개, KO: {len(ko_docs):,}개")
    
    # 3. Nodes 생성
    print("\n[4/7] Nodes 생성 중...")
    cn_nodes = build_nodes(cn_docs)
    ko_nodes = build_nodes(ko_docs)
    print(f"✅ CN: {len(cn_nodes):,}개, KO: {len(ko_nodes):,}개")
    
    # CN/KO 별도 컬렉션 초기화 (clear 옵션 전달)
    print("\n[5/7] Vector Store 초기화 중...")
    cn_store, ko_store = init_chroma(persist_dir, clear=clear)
    if clear:
        print("✅ 기존 컬렉션 삭제 후 재생성")
    else:
        print("✅ 기존 컬렉션 로드(증분 Indexing 모드)")
    
    # 5. CN 인덱싱
    print("\n[6/7] CN 인덱싱 중...")
    logger.info("[indexing] Indexing CN collection...")
    _ = build_index_from_nodes(cn_nodes, cn_store)
    
    print("\n[6/7] KO 인덱싱 중...")
    logger.info("[indexing] Indexing KO collection...")
    _ = build_index_from_nodes(ko_nodes, ko_store)
    
    # 캐시 저장 (BM25용)
    print("\n[7/7] 캐시 저장 중...")
    if clear:
        # 전체 ReIndexing: Cache 덮어쓰기
        save_nodes_cache(cn_nodes, ko_nodes)
        print("✅ Cache 저장 완료 (전체 재생성)")
    else:
        # 증분 Indexing: 기존 캐시에 병합
        save_nodes_cache_incremental(cn_nodes, ko_nodes)
        print("✅ Cache 업데이트 완료 (증분 Indexing)")

    # === 완료 ===
    print("\n" + "="*60)
    print("🎉 인덱싱 완료!")
    print("="*60)
    print(f"📊 통계:")
    print(f"  - CSV 행: {len(df):,}개")
    print(f"  - CN 문서: {len(cn_docs):,}개 → {len(cn_nodes):,}개 노드")
    print(f"  - KO 문서: {len(ko_docs):,}개 → {len(ko_nodes):,}개 노드")
            
    logger.info(
        "[indexing] finished. rows=%d, cn_docs=%d, ko_docs=%d, cn_nodes=%d, ko_nodes=%d, mode=%s",
        len(df), len(cn_docs), len(ko_docs), len(cn_nodes), len(ko_nodes),
        "full" if clear else "incremental"
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
