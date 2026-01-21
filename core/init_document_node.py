# core/init_document_node.py
# Document, Node 생성 및 캐시 관리

import os
import pandas as pd
import pickle

from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser

from core.normalizer import (
    to_embedding_normalize_for_search,
    split_text_segments,
    split_segments_paired,
)
from core.config import Settings
from utils import get_translator_logger


logger = get_translator_logger("core.init_document_node")


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
    
    Returns:
        (cn_docs, ko_docs) 튜플
        - cn_docs: CN 컬렉션용 (cn, cn_normalized, cn_segment)
        - ko_docs: KO 컬렉션용 (ko, ko_segment)
    """
    cn_docs: list[Document] = []
    ko_docs: list[Document] = []
    
    for i, row in df.iterrows():
        cn = row["cn"]
        ko = row["ko"]
        base_id = int(i) + 1
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
                    doc_id=f"cn_segment-{base_id}-{j}",
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
                    doc_id=f"ko_segment-{base_id}-{j}",
                    metadata=ko_seg_meta
                ))
        else:
            # 매핑 실패: CN segment만 생성
            segments = split_text_segments(cn)
            for j, segment in enumerate(segments):
                if segment == cn:
                    continue
                
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
                    doc_id=f"cn_segment-{base_id}-{j}",
                    metadata=segment_meta
                ))
    
    logger.info(
        "Documents built: cn_docs=%d, ko_docs=%d (from %d rows)",
        len(cn_docs), len(ko_docs), len(df)
    )
    return cn_docs, ko_docs


# ===============================================
# Node 생성 및 캐시 관리
# ===============================================

def build_nodes(docs: list[Document]):
    """
    문서를 Node(검색 단위 블록)로 변환.
    """
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
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
