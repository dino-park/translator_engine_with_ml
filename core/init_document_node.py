# core/init_document_node.py
# Document, Node 생성 및 캐시 관리

import os
import pandas as pd
import pickle
import hashlib
from tqdm import tqdm

from llama_index.core import Document

from core.normalizer import (
    to_embedding_normalize_for_search,
    split_text_segments,
    split_segments_paired,
)
from core.config import Settings
from utils import get_translator_logger


logger = get_translator_logger("core.init_document_node")


def generate_id_from_text(text: str) -> str:
    """
    Text에서 고유 ID 생성(Hash 기반)
    - 동일 텍스트는 항상 동일 ID 반환
    - 충돌 없이 증분 Indexing 가능

    Args:
        text (str): 고유 ID 생성할 텍스트

    Returns:
        str: 8자리 Hash ID
    """
    # SHA256 Hash 처음 8자리 사용
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    csv 로드 및 기본 정리
    - 필수 컬럼: cn, ko

    Args:
        csv_path (str): 파일 경로

    Returns:
        pd.DataFrame: 
    """
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    # ----- 컬럼명 소문자로 통일 -----
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    # ----- cn key 기준으로, 중복된 행을 제거하고 첫 번째 행만 유지
    df = df[["cn", "ko"]].drop_duplicates(subset="cn", keep="first")
    # ----- NaN/공백 정리 -----
    df[["cn", "ko"]] = df[["cn", "ko"]].fillna("").astype(str)
    df["cn"] = df["cn"].str.strip()
    df["ko"] = df["ko"].str.strip()
    
    df = df[df["cn"] != ""]
    return df


def build_documents(df: pd.DataFrame) -> tuple[list[Document], list[Document]]:
    """
    CN/KO 별도 컬렉션용 Documents 생성
    - 콘텐츠 기반 ID 생성 (Hash) -> 증분 Indexing 지원
    - 메타데이터가 너무 긴 경우(20자 초과) skip
    
    Returns:
        (cn_docs, ko_docs) 튜플
        - cn_docs: CN 컬렉션용 (cn, cn_normalized, cn_segment)
        - ko_docs: KO 컬렉션용 (ko, ko_segment)
    """
    # 메타데이터 최대 길이 제한 (chunk_size보다 작아야 함)
    # cn, ko 텍스트가 메타데이터의 대부분을 차지하므로 이들만 체크
    MAX_METADATA_LENGTH = 20  # 20자 초과는 비정상 데이터로 간주
    
    cn_docs: list[Document] = []
    ko_docs: list[Document] = []
    skipped_count = 0
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="📄 Documents", unit="rows", ncols=80):
        cn = row["cn"]
        ko = row["ko"]
        
        # ===== 메타데이터 길이 사전 체크 (너무 긴 경우 skip) =====
        # cn, ko 텍스트가 개별적으로 너무 길면 해당 행 전체 skip
        if len(cn) > MAX_METADATA_LENGTH or (ko and len(ko) > MAX_METADATA_LENGTH):
            skipped_count += 1
            logger.warning(
                f"Skipping row {i}: text too long (cn={len(cn)}, ko={len(ko) if ko else 0}, max={MAX_METADATA_LENGTH})"
            )
            continue
        
        # ===== 콘텐츠 기반 ID 생성(충돌 방지) =====
        base_id = generate_id_from_text(cn)
        entry_id = f"entry-{base_id}"
        
        # ===== CN Documents =====
        # CN 원본
        cn_meta = {
            "entry_id": entry_id,
            "cn": cn,
            "doc_type": "cn"
        }
        if ko:
            cn_meta["ko"] = ko
        cn_docs.append(Document(text=cn, doc_id=f"cn-{base_id}", metadata=cn_meta))
        
        # CN normalized
        cn_normalized = to_embedding_normalize_for_search(cn)
        if cn_normalized and cn_normalized != cn:
            cn_normalized_meta = {
                "entry_id": entry_id,
                "cn": cn,
                "doc_type": "cn_normalized"
            }
            if ko:
                cn_normalized_meta["ko"] = ko
            cn_docs.append(Document(
                text=cn_normalized,
                doc_id=f"cn_normalized-{base_id}",
                metadata=cn_normalized_meta
            ))
        
        # ===== KO Documents =====
        if ko:
            ko_meta = {
                "entry_id": entry_id,
                "cn": cn,
                "ko": ko,
                "doc_type": "ko"
            }
            ko_docs.append(Document(text=ko, doc_id=f"ko-{base_id}", metadata=ko_meta))
        
        # ===== Segment Documents =====
        paired_segments = split_segments_paired(cn, ko) if ko else []
        
        if paired_segments:
            for j, (cn_seg, ko_seg) in enumerate(paired_segments):
                if cn_seg == cn:
                    continue
                
                # Segment 길이 체크 (20자 초과 시 skip)
                if len(cn_seg) > MAX_METADATA_LENGTH or (ko_seg and len(ko_seg) > MAX_METADATA_LENGTH):
                    logger.debug(f"Skipping segment: cn_seg={len(cn_seg)}, ko_seg={len(ko_seg) if ko_seg else 0}")
                    continue
                
                seg_id = generate_id_from_text(f"{cn}_{cn_seg}_{j}")
                segment_entry_id = f"{entry_id}-seg-{j}"
                
                # CN segment → cn_docs
                cn_seg_meta = {
                    "entry_id": segment_entry_id,
                    "parent_entry_id": entry_id,
                    "cn": cn_seg,
                    "ko": ko_seg,
                    "parent_cn": cn,
                    "parent_ko": ko,
                    "doc_type": "cn_segment",
                    "segment_index": j,
                }
                cn_docs.append(Document(
                    text=cn_seg,
                    doc_id=f"cn_segment-{seg_id}",
                    metadata=cn_seg_meta
                ))
                
                # KO segment → ko_docs
                ko_seg_meta = {
                    "entry_id": segment_entry_id,
                    "parent_entry_id": entry_id,
                    "cn": cn_seg,
                    "ko": ko_seg,
                    "parent_cn": cn,
                    "parent_ko": ko,
                    "doc_type": "ko_segment",
                    "segment_index": j,
                }
                ko_docs.append(Document(
                    text=ko_seg,
                    doc_id=f"ko_segment-{seg_id}",
                    metadata=ko_seg_meta
                ))
        else:
            # 매핑 실패: CN segment만 생성
            segments = split_text_segments(cn)
            for j, segment in enumerate(segments):
                if segment == cn:
                    continue
                
                # Segment 길이 체크 (20자 초과 시 skip)
                if len(segment) > MAX_METADATA_LENGTH:
                    logger.debug(f"Skipping CN-only segment: len={len(segment)}")
                    continue
                
                seg_id = generate_id_from_text(f"{cn}_{segment}_{j}")
                segment_entry_id = f"{entry_id}-seg-{j}"
                segment_meta = {
                    "entry_id": segment_entry_id,
                    "parent_entry_id": entry_id,
                    "cn": segment,
                    "parent_cn": cn,
                    "doc_type": "cn_segment",
                    "segment_index": j,
                }
                if ko:
                    segment_meta["parent_ko"] = ko
                
                cn_docs.append(Document(
                    text=segment,
                    doc_id=f"cn_segment-{seg_id}",
                    metadata=segment_meta
                ))
    
    logger.info(
        "Documents built: cn_docs=%d, ko_docs=%d (from %d rows, skipped %d rows with long text)",
        len(cn_docs), len(ko_docs), len(df), skipped_count
    )
    if skipped_count > 0:
        print(f"⚠️  {skipped_count}개 행이 텍스트 길이 초과로 skip되었습니다 (최대: {MAX_METADATA_LENGTH}자)")
    return cn_docs, ko_docs


# ===============================================
# Node 생성 및 캐시 관리
# ===============================================

def build_nodes(docs: list[Document]):
    """
    문서를 Node(검색 단위 블록)로 변환.
    - chunk_size=512: MAX_METADATA_LENGTH(20자)에 맞춰 여유있게 설정
    - 20자 이하 텍스트만 들어오므로 512면 충분
    """
    from llama_index.core.node_parser import SentenceSplitter
    parser = SentenceSplitter(
        chunk_size=512,         # 100자 텍스트 + 메타데이터 여유분
        chunk_overlap=50,
    )
    nodes = parser.get_nodes_from_documents(docs)
    logger.info("Nodes built: %d nodes from %d documents", len(nodes), len(docs))
    return nodes


def _get_cache_path(suffix: str = "") -> str:
    """nodes 캐시 파일 경로 반환 (내부용)"""
    persist_dir = Settings.env.get("PERSIST_DIR", ".")
    filename = f"nodes_cache{suffix}.pkl"
    return os.path.join(persist_dir, filename)


def _save_cache(nodes, suffix: str = "") -> None:
    """nodes를 pickle 파일로 저장 (내부용)"""
    cache_path = _get_cache_path(suffix)
    with open(cache_path, "wb") as f:
        pickle.dump(nodes, f)
    logger.info("Nodes cached to %s", cache_path)


def _load_cache(suffix: str = ""):
    """캐시된 nodes 로드 (내부용). 없으면 None 반환"""
    cache_path = _get_cache_path(suffix)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            nodes = pickle.load(f)
        logger.info("Nodes loaded from cache: %s (%d nodes)", cache_path, len(nodes))
        return nodes
    logger.info("No cache found for suffix=%r", suffix)
    return None


def save_nodes_cache(cn_nodes, ko_nodes) -> None:
    """CN/KO nodes를 각각 캐시 저장"""
    _save_cache(cn_nodes, "_cn")
    _save_cache(ko_nodes, "_ko")


def load_nodes_cache() -> tuple:
    """
    CN/KO nodes 캐시 로드
    
    Returns:
        (cn_nodes, ko_nodes) 또는 (None, None)
    """
    cn_nodes = _load_cache("_cn")
    ko_nodes = _load_cache("_ko")
    if cn_nodes is None or ko_nodes is None:
        return None, None
    return cn_nodes, ko_nodes


# core/init_document_node.py 하단에 추가

def save_nodes_cache_incremental(cn_nodes, ko_nodes) -> None:
    """
    CN/KO nodes를 기존 캐시에 병합하여 저장 (증분 업데이트)
    - 같은 node_id는 덮어쓰기 (업데이트)
    - 새로운 node_id는 추가
    
    Args:
        cn_nodes: 새로 추가할 CN nodes
        ko_nodes: 새로 추가할 KO nodes
    
    사용 시나리오:
        - 신규 CSV 파일 추가 시
        - 기존 CSV 파일 수정 시
    """
    # 기존 캐시 로드
    existing_cn, existing_ko = load_nodes_cache()
    
    if existing_cn and existing_ko:
        logger.info("Merging with existing cache: cn=%d, ko=%d", len(existing_cn), len(existing_ko))
        
        # dict로 변환하여 병합 (node_id 기준)
        cn_dict = {n.node_id: n for n in existing_cn}
        ko_dict = {n.node_id: n for n in existing_ko}
        
        # 새 nodes 추가/업데이트
        updated_cn = 0
        added_cn = 0
        for n in cn_nodes:
            if n.node_id in cn_dict:
                updated_cn += 1
            else:
                added_cn += 1
            cn_dict[n.node_id] = n
        
        updated_ko = 0
        added_ko = 0
        for n in ko_nodes:
            if n.node_id in ko_dict:
                updated_ko += 1
            else:
                added_ko += 1
            ko_dict[n.node_id] = n
        
        # 캐시 저장
        _save_cache(list(cn_dict.values()), "_cn")
        _save_cache(list(ko_dict.values()), "_ko")
        
        logger.info(
            "Cache updated: CN (added=%d, updated=%d, total=%d), KO (added=%d, updated=%d, total=%d)",
            added_cn, updated_cn, len(cn_dict),
            added_ko, updated_ko, len(ko_dict)
        )
        print(f"📦 캐시 업데이트:")
        print(f"  - CN: 추가 {added_cn}개, 업데이트 {updated_cn}개, 총 {len(cn_dict)}개")
        print(f"  - KO: 추가 {added_ko}개, 업데이트 {updated_ko}개, 총 {len(ko_dict)}개")
    else:
        # 기존 캐시 없으면 새로 저장
        logger.info("No existing cache, creating new cache")
        _save_cache(cn_nodes, "_cn")
        _save_cache(ko_nodes, "_ko")
        print(f"📦 새 캐시 생성: CN {len(cn_nodes)}개, KO {len(ko_nodes)}개")
