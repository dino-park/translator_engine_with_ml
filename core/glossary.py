# core/glossary.py
# Glossary 관리(segment 부분 매칭용 포함)

import jieba
import glob
import pandas as pd
from pathlib import Path

from core.init_document_node import load_dataset
from core.normalizer import normalize_for_exact_match
from core.retriever import QueryFusionRetriever
from core.config import SCORE_THRESHOLD
from utils import get_translator_logger

logger = get_translator_logger("core.glossary")


# -----------------------------
# Glossary Map 캐싱 (부분 매칭용)
# -----------------------------

# Glossary 디렉토리 경로
GLOSSARY_DIR = "glossary"


def _load_all_glossary_data() -> pd.DataFrame:
    """
    Glossary 디렉토리의 모든 CSV 파일을 로드하여 병합
    - indexing.py의 load_all_csvs_from_directory() 함수와 동일한 기능

    Returns:
        pd.DataFrame: 병합된 DataFrame
    """
    csv_files = sorted(glob.glob(f"{GLOSSARY_DIR}/*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {GLOSSARY_DIR}")
        return pd.DataFrame(columns=["cn", "ko"])
    
    logger.info(f"Loading {len(csv_files)} CSV files for glossary map")
    
    all_dfs = []
    for csv_path in csv_files:
        try:
            df = load_dataset(csv_path)
            all_dfs.append(df)
            logger.info(f"  ✓ {Path(csv_path).name}: {len(df)} rows")
        except Exception as e:
            logger.error(f"   ✗ Failed to load {Path(csv_path).name}: {e}")
    
    if not all_dfs:
        logger.warning("No valid CSV files loaded")
        return pd.DataFrame(columns=["cn", "ko"])
    
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
    
    logger.info(f"Total glossary entries: {after_count}")
    return merged
    

def build_glossary_map() -> dict[str, tuple[str, str]]:
    """
    모든 CSV에서 glossary_map을 생성 (jieba로 segment도 포함)
    
    - lookup_glossary_by_segments에서 사용
    - Document/Node 증가 없이 부분 매칭 지원
    - 초기화 시 한 번만 생성
    
    Args:
        csv_path: glossary CSV 파일 경로 (기본: BASE_CSV_PATH)
    
    Returns:
        {정규화된_cn: (원본_cn, ko, is_segment)} 딕셔너리
        is_segment: True면 jieba로 분해된 segment
    
    Note:
        원본 항목(is_segment=False)이 segment(is_segment=True)보다 항상 우선함.
        즉, 荔倾이 원본으로 등록되면, 荔倾的门匙에서 추출한 segment는 덮어쓰지 않음.
    """
    
    df = _load_all_glossary_data()
    
    glossary_map: dict[str, tuple[str, str, bool]] = {}
    
    # 1단계: 모든 원본 항목을 먼저 추가 (is_segment=False)
    for _, row in df.iterrows():
        cn = row.get("cn", "")
        ko = row.get("ko", "")
        
        if not cn or not ko:
            continue
        
        normalized_cn = normalize_for_exact_match(cn)
        if normalized_cn not in glossary_map:
            glossary_map[normalized_cn] = (cn, ko, False)
    
    # 2단계: segment 추가 (원본이 없는 경우에만)
    for _, row in df.iterrows():
        cn = row.get("cn", "")
        ko = row.get("ko", "")
        
        if not cn or not ko:
            continue
        
        # jieba로 분해한 segment 추가
        segments = list(jieba.cut(cn))
        for seg in segments:
            if len(seg) < 2:
                continue
            # segment가 원본과 같으면 스킵
            if seg == cn:
                continue
            
            normalized_seg = normalize_for_exact_match(seg)
            # 원본 항목(is_segment=False)이 이미 있으면 덮어쓰지 않음
            if normalized_seg not in glossary_map:
                # segment의 ko는 parent의 ko (LLM이 문맥 판단)
                glossary_map[normalized_seg] = (seg, ko, True)
    
    logger.info("Glossary map built: %d entries (including segments)", len(glossary_map))
    return glossary_map


def build_glossary_map_ko() -> dict[str, tuple[str, str]]:
    """
    한국어 기준 glossary_map 생성 (역방향 lookup 용도)
    
    Returns:
        {normalized_ko: (원본_ko, cn)} 딕셔너리
    """
    
    df = _load_all_glossary_data()
    
    glossary_map_ko: dict[str, tuple[str, str, bool]] = {}
    
    for _, row in df.iterrows():
        cn = row.get("cn", "")
        ko = row.get("ko", "")
        
        if not cn or not ko:
            continue
        
        # 한국어 정규화
        normalized_ko = normalize_for_exact_match(ko)
        if normalized_ko not in glossary_map_ko:
            glossary_map_ko[normalized_ko] = (ko, cn)
    
    logger.info("Glossary map (ko) build: %d entries", len(glossary_map_ko))
    return glossary_map_ko


# ===============================================
# Glossary-aware Sentence Translation (용어 기반 문장 번역)
# ===============================================

def lookup_glossary(
    terms: list[str],
    hybrid: "QueryFusionRetriever",
    src_lang: str,
    max_hybrid_calls: int = 5
) -> dict[str, str]:
    """
    VectorDB에서 용어들을 일괄 검색하여 매칭된 용어-번역 쌍 반환
    
    Args:
        terms: 용어 후보 리스트 (긴 것부터 정렬되어 있어야 함)
        hybrid: QueryFusionRetriever 인스턴스
        src_lang: 원본 언어 ("cn" 또는 "ko")
        max_hybrid_calls: hybrid 검색 최대 호출 횟수 (API 비용 제한, 기본값 5)
    
    Returns:
        {원본 용어: 번역된 용어} 딕셔너리
    """
    glossary = {}
    matched_terms = set()  # 이미 매칭된 용어 (하위 n-gram 중복 방지)
    hybrid_calls = 0
    
    for term in terms:
        # 이미 정답 용어가 매칭되었을 경우, 다음 단어가 더 긴 용어에 포함되면 스킵 (하위 n-gram 중복 방지)
        if any(term in matched for matched in matched_terms):
            continue
        
        # 최대 hybrid 호출 횟수 도달 시 종료
        if hybrid_calls >= max_hybrid_calls:
            logger.info(
                "Reached max_hybrid_calls=%d, stopping. Remaining terms skipped.",
                max_hybrid_calls
            )
            break
        
        try:
            results = hybrid.retrieve(term)
            hybrid_calls += 1
            if not results:
                continue
            
            top = results[0]
            score = getattr(top, "score", None)
            md = top.node.metadata or {}
            top_cn = md.get("cn", "")
            top_ko = md.get("ko", "")
            
            # 정확 매칭 또는 score threshold 이상
            is_exact_match = (top_cn == term) or (top_ko == term)
            is_reliable = is_exact_match or (score is not None and score >= SCORE_THRESHOLD)
            
            if is_reliable:
                logger.info(
                    "Checking: term=%r, top_cn=%r, top_ko=%r, src_lang=%s, "
                    "top_cn==term=%s, bool(top_ko)=%s",
                    term, top_cn, top_ko, src_lang, top_cn == term, bool(top_ko)
                )
                
                # 중국어 검색: 정확 매칭만
                if src_lang == "cn" and top_cn == term and top_ko:
                    glossary[term] = top_ko
                    matched_terms.add(term)
                    logger.debug("Glossary match (cn): %r → %r (score=%s)", term, top_ko, score)
                
                # 한국어 검색: 정확 매칭 또는 부분 매칭 (top_ko가 term에 포함될 때)
                elif src_lang == "ko" and top_cn:
                    is_ko_exact = (top_ko == term)
                    # 부분 매칭: top_ko가 term에 포함되고, top_ko가 2글자 이상
                    is_ko_partial = (top_ko and len(top_ko) >= 2 and top_ko in term)
                    
                    if is_ko_exact or is_ko_partial:
                        glossary[term] = top_cn
                        matched_terms.add(term)
                        match_type = "exact" if is_ko_exact else "partial"  # 정확 매칭 또는 부분 매칭 구분
                        logger.debug("Glossary match (ko, %s): %r → %r (score=%s)", match_type, term, top_cn, score)
        
        except Exception as e:
            logger.warning("Glossary lookup failed for %r: %s", term, e)
            continue
    
    return glossary


def lookup_glossary_in_sentence(
    sentence: str,
    glossary_map_ko: dict[str, tuple[str, str]],
    min_term_len: int = 2
) -> dict[str, str]:
    """
    문장에서 Glossary ko가 포함되어 있는지 역방향 검색
    - 캐시된 ko 용어들을 문장에서 직접 찾음
    - vectorDB 호출 없이 메모리에서 O(N) 스캔
    Args:
        sentence (str): 검색할 문장
        glossary_map_ko: build_glossary_map_ko()으로 생성된 딕셔너리
        min_term_len (int, optional): 최소 용어 길이 (기본: 2)
    Returns:
        {원본_ko: 번역된_cn} 딕셔너리
    """
    
    found = {}
    
    # 문장 정규화(비교용)
    normalized_sentence = normalize_for_exact_match(sentence)
    
    for normalized_ko, (original_ko, cn) in glossary_map_ko.items():
        # 최소 길이 체크(오탐 방지용)
        if len(original_ko) < min_term_len:
            continue
        # 정규화된 ko가 정규화된 문장에 포함되는지 체크
        if normalized_ko in normalized_sentence:
            # 이미 매칭된 용어인지 체크, 더 긴 용어가 있으면 스킵
            existing = found.get(original_ko)
            if not existing:
                found[original_ko] = cn
                logger.debug("lookup_glossary_in_senetence: found %r -> %r", original_ko, cn)
                
    final = {}
    sorted_keys = sorted(found.keys(), key=len, reverse=True)
    matched_positions = set()
    
    # 긴 용어부터 처리, 중복 방지를 위해 이미 처리된 위치 체크
    for ko in sorted_keys:
        start = sentence.find(ko)
        if start == -1:
            continue
        end = start + len(ko)
        
        overlaps = any(
            not (end <= pos[0] or start >= pos[1])
            for pos in matched_positions
        )
        if not overlaps:
            final[ko] = found[ko]
            matched_positions.add((start, end))
    
    if final:
        logger.info("lookup_glossary_in_sentence: %d matches - %s", len(final), final)
    
    return final
    

def lookup_glossary_by_segments(
    query: str, 
    glossary_map: dict[str, tuple[str, str, bool]], 
    src_lang: str = "cn",
    candidates: list = None,
    bm25_retriever = None
) -> dict | None:
    """
    쿼리를 jieba로 분해하여 각 세그먼트가 glossary에 있는지 확인.
    LLM fallback 시 glossary 힌트로 활용.
    
    Args:
        query: 검색 쿼리 (예: "全联堂场景赠送")
        glossary_map: build_glossary_map()으로 생성된 딕셔너리
                      {정규화된_cn: (원본_cn, ko, is_segment)}
        src_lang: 원본 언어 ("cn" 또는 "ko")
        candidates: translate_term에서 이미 검색된 VectorDB 결과 (추가 API 호출 없이 재사용)
        bm25_retriever: BM25 retriever (suffix 검색용, 로컬 동작 - API 호출 없음)
    
    Returns:
        모든 매칭 정보 반환:
        {
            "matches": [{"cn": "全联堂", "ko": "전연당", "is_segment": False, "exact_ko": "전연당"}, ...],
            "all_segments": ["全联堂", "场景", "赠送"],
            "unmatched_segments": ["场景", "赠送"],
            "similar_patterns": [{"pattern": "鹿游角色单", "translation": "로쿠유 캐릭터 목록"}, ...]
        }
        매칭이 없으면 None
        
    Example:
        query = "全联堂场景赠送"
        → {"matches": [{"cn": "全联堂", "ko": "전연당", "is_segment": False, "exact_ko": "전연당"}], 
           "all_segments": ["全联堂", "场景", "赠送"],
           "unmatched_segments": ["场景", "赠送"],
           "similar_patterns": []}
    """
    # 쿼리를 jieba로 분해
    segments = list(jieba.cut(query))
    logger.debug("lookup_glossary_by_segments: query=%r, segments=%r", query, segments)
    
    # 각 세그먼트가 glossary에 있는지 확인
    matches = []
    unmatched_segments = []
    
    for segment in segments:
        # 단일 문자는 건너뜀
        if len(segment) <= 1:
            continue
            
        normalized_segment = normalize_for_exact_match(segment)
        if normalized_segment in glossary_map:
            original_cn, ko, is_segment = glossary_map[normalized_segment]
            
            # is_segment=True인 경우, 원본 cn의 정확한 ko 번역을 별도로 조회
            exact_ko = None
            if is_segment:
                # glossary_map 전체를 순회하여 segment와 동일한 cn이면서 is_segment=False인 항목 찾기
                for norm_key, (orig_cn, orig_ko, orig_is_seg) in glossary_map.items():
                    if orig_cn == segment and not orig_is_seg:
                        exact_ko = orig_ko
                        logger.debug("lookup_glossary_by_segments: found exact_ko for segment=%r: %r", segment, exact_ko)
                        break
            else:
                exact_ko = ko  # is_segment=False면 ko 자체가 정확한 번역
            
            matches.append({
                "cn": original_cn, 
                "ko": ko, 
                "segment": segment, 
                "is_segment": is_segment,
                "exact_ko": exact_ko  # 정확한 번역 (있으면)
            })
            logger.info("lookup_glossary_by_segments: found segment=%r, cn=%r, ko=%r, is_segment=%s, exact_ko=%r", 
                       segment, original_cn, ko, is_segment, exact_ko)
        else:
            unmatched_segments.append(segment)
    
    if not matches:
        logger.debug("lookup_glossary_by_segments: no matching segment found")
        return None
    
    # 유사 suffix 패턴 검색 (마지막 segment로 끝나는 기존 번역 찾기)
    similar_patterns = []
    if len(segments) >= 2:
        # 마지막 2~3개 segment를 suffix로 사용
        suffix_candidates = []
        if len(segments) >= 2:
            suffix_candidates.append("".join(segments[-2:]))  # 마지막 2개
        if len(segments) >= 3:
            suffix_candidates.append("".join(segments[-3:]))  # 마지막 3개
        
        for suffix in suffix_candidates:
            if len(suffix) < 2:
                continue
            # 1. glossary_map(CSV 기반)에서 같은 suffix로 끝나는 다른 항목 찾기
            for norm_key, (orig_cn, orig_ko, orig_is_seg) in glossary_map.items():
                if orig_cn != query and orig_cn.endswith(suffix) and not orig_is_seg:
                    similar_patterns.append({
                        "pattern": orig_cn,
                        "translation": orig_ko,
                        "suffix": suffix,
                        "source": "csv"
                    })
                    logger.debug("lookup_glossary_by_segments: found similar pattern (csv)=%r → %r (suffix=%r)", 
                               orig_cn, orig_ko, suffix)
                    if len(similar_patterns) >= 3:  # 최대 3개
                        break
            
            # 2. 이미 검색된 candidates에서 suffix 매칭 (추가 API 호출 없음)
            if len(similar_patterns) < 3 and candidates:
                for r in candidates:
                    # candidates는 dict 형태 (format_candidates 결과)
                    r_cn = r.get("cn", "")
                    r_ko = r.get("ko", "")
                    doc_type = r.get("matched_by", "")
                    
                    # 자신과 다르고, suffix로 끝나는 항목만
                    if r_cn and r_ko and r_cn != query and r_cn.endswith(suffix):
                        # 이미 추가된 패턴인지 확인
                        if not any(p["pattern"] == r_cn for p in similar_patterns):
                            similar_patterns.append({
                                "pattern": r_cn,
                                "translation": r_ko,
                                "suffix": suffix,
                                "source": doc_type or "candidates"
                            })
                            logger.debug("lookup_glossary_by_segments: found similar pattern (candidates)=%r → %r (suffix=%r)", 
                                       r_cn, r_ko, suffix)
                            if len(similar_patterns) >= 3:
                                break
            
            # 3. user_term 메모리 캐시에서 suffix 검색 (실시간 반영, API 호출 없음)
            # - 이번 세션에서 추가된 user_term도 즉시 검색 가능
            if len(similar_patterns) < 3:
                try:
                    from core.chroma_utils import get_user_term_cache
                    user_term_cache = get_user_term_cache()
                    for entry in user_term_cache:
                        r_cn = entry.get("cn", "")
                        r_ko = entry.get("ko", "")
                        doc_type = entry.get("doc_type", "")
                        
                        # 자신과 다르고, suffix로 끝나는 항목만
                        if r_cn and r_ko and r_cn != query and r_cn.endswith(suffix):
                            # 이미 추가된 패턴인지 확인
                            if not any(p["pattern"] == r_cn for p in similar_patterns):
                                similar_patterns.append({
                                    "pattern": r_cn,
                                    "translation": r_ko,
                                    "suffix": suffix,
                                    "source": f"user_term_cache:{doc_type}"
                                })
                                logger.info("lookup_glossary_by_segments: found similar pattern (cache)=%r → %r (suffix=%r)", 
                                           r_cn, r_ko, suffix)
                                if len(similar_patterns) >= 3:
                                    break
                except Exception as e:
                    logger.warning("user_term cache search failed: %s", e)
            
            # 4. BM25로 suffix 검색 (앱 시작 시 빌드된 nodes 기준, API 호출 없음)
            if len(similar_patterns) < 3 and bm25_retriever:
                try:
                    bm25_results = bm25_retriever.retrieve(suffix)
                    for r in bm25_results[:10]:  # 상위 10개 확인
                        md = r.node.metadata or {}
                        r_cn = md.get("cn", "")
                        r_ko = md.get("ko", "")
                        doc_type = md.get("doc_type", "")
                        
                        # 자신과 다르고, suffix로 끝나는 항목만
                        if r_cn and r_ko and r_cn != query and r_cn.endswith(suffix):
                            # 이미 추가된 패턴인지 확인
                            if not any(p["pattern"] == r_cn for p in similar_patterns):
                                similar_patterns.append({
                                    "pattern": r_cn,
                                    "translation": r_ko,
                                    "suffix": suffix,
                                    "source": f"bm25:{doc_type}" if doc_type else "bm25"
                                })
                                logger.info("lookup_glossary_by_segments: found similar pattern (bm25)=%r → %r (suffix=%r, doc_type=%r)", 
                                           r_cn, r_ko, suffix, doc_type)
                                if len(similar_patterns) >= 3:
                                    break
                except Exception as e:
                    logger.warning("BM25 suffix search failed: %s", e)
            
            if similar_patterns:
                break  # 찾으면 더 이상 검색 안 함
    
    return {
        "matches": matches,
        "all_segments": segments,
        "unmatched_segments": unmatched_segments,
        "similar_patterns": similar_patterns,
    }
