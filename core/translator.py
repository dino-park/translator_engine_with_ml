# core/translator.py
# 번역 엔진 핵심 로직

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from core.retriever import search, format_candidates
from core.normalizer import normalize_for_exact_match
from core.glossary import lookup_glossary_by_segments, lookup_glossary, lookup_glossary_in_sentence, build_glossary_map_ko
from core.chroma_utils import add_user_entry_to_glossary
from core.config import Settings, TranslationReason, SCORE_THRESHOLD
from preprocessor import extract_terms_from_sentence, detect_mixed_language
from utils import get_translator_logger
from preprocessor import normalize_special_symbols

logger = get_translator_logger("core.translator")

# ===== 한국어 Glossary Map 캐싱 (역방향 lookup용) =====
_glossary_map_ko = None

def _get_glossary_map_ko():
    """Lazy loading으로 한국어 glossary map 반환"""
    global _glossary_map_ko
    if _glossary_map_ko is None:
        try:
            _glossary_map_ko = build_glossary_map_ko()
            logger.info("Loaded glossary_map_ko: %d entries", len(_glossary_map_ko))
        except Exception as e:
            logger.error("Failed to build glossary_map_ko: %s", e, exc_info=True)
            _glossary_map_ko = {}  # 빈 딕셔너리로 fallback
    return _glossary_map_ko

# 정확 매칭 최소 점수 임계값
# - 정확 매칭이라도 점수가 너무 낮으면 신뢰하지 않음 (오염된 데이터 방지)
# - LLM fallback으로 추가된 잘못된 데이터는 보통 점수가 낮음
EXACT_MATCH_MIN_SCORE = 0.02  # Hybrid Search의 Reciprocal Rank Fusion score 범위 고려

# === 1단계 대응: 신뢰도 기반 필터링 상수 ===
# 1위-2위 점수 차이 최소값 (동점이면 확정 금지)
CANDIDATE_GAP_THRESHOLD = 0.0005

def translate_term(
    query: str,
    cn_retriever: QueryFusionRetriever,
    ko_retriever: QueryFusionRetriever,
    src_lang: str,
    use_llm_fallback: bool = True,
    cn_bm25_retriever: BM25Retriever = None,
    ko_bm25_retriever: BM25Retriever = None,
    glossary_map: dict[str, tuple[str, str]] = None
    ) -> dict:
    """
    - 양방향 hybrid 검색 후 병합 후 정확 매칭 찾기
    
    Args:
        query: 검색 쿼리
        cn_retriever: CN 컬렉션 QueryFusionRetriever (Vector + BM25 통합)
        ko_retriever: KO 컬렉션 QueryFusionRetriever (Vector + BM25 통합)
        src_lang: 원본 언어 ("cn" 또는 "ko")
        use_llm_fallback: LLM fallback 사용 여부
        cn_bm25_retriever: CN BM25 전용 retriever (로깅/분석용)
        ko_bm25_retriever: KO BM25 전용 retriever (로깅/분석용)
        glossary_map: build_glossary_map()으로 생성된 딕셔너리 (부분 매칭용)
    """
    logger.info("translate_term called, query=%r, src_lang=%s", query, src_lang)
    
    # 정규화는 한 번만 수행 (lru_cache로 캐싱됨)
    normalized_query = normalize_for_exact_match(query)
    
    # ===== 1단계: 단방향 검색 =====
    merged_results = search(
        query, cn_retriever, ko_retriever, 
        src_lang=src_lang, top_k=5,
        fallback_to_two_way=False
    )
    logger.info("search returned %d merged results (src_lang=%s)", len(merged_results), src_lang)
    
    # ===== 양방향 Fallback 조건 체크 =====
    should_retry_two_way = False
    retry_reason = None
    
    # Condition 1: 결과 없음
    if len(merged_results) == 0:
        should_retry_two_way = True
        retry_reason = "no_results"
        
    # Condition 2: top_score가 너무 낮음
    elif merged_results[0].get("score", 0) < SCORE_THRESHOLD:
        should_retry_two_way = True
        retry_reason = "low_top_score"
        
    # Condition 3: candidate_gap이 너무 작음 (동점)
    elif len(merged_results) >= 2:
        candidate_gap = merged_results[0]["score"] - merged_results[1]["score"]
        if candidate_gap < CANDIDATE_GAP_THRESHOLD:
            should_retry_two_way = True
            retry_reason = f"small_gap({candidate_gap:.6f})"
    
    # Condition 4: exact_match가 없음 + BM25도 확신 없음
    # (BM25 분석은 아래에서 수행되므로 그 이후에 체크함)
    
    # Condition 5: 혼합 언어 감지
    if not should_retry_two_way and detect_mixed_language(query):
        should_retry_two_way = True
        retry_reason = "mixed_language_detected"
    
    # ===== 2단계: 위 조건 충족 시 양방향 재검색 =====
    if should_retry_two_way:
        logger.info("Retrying with two-way search (reason=%s)", retry_reason)
        merged_results = search(
            query, cn_retriever, ko_retriever,
            src_lang=src_lang, top_k=5,
            fallback_to_two_way=True
        )
        logger.info("Two-way search returned %d results", len(merged_results))
    
    # 후보 포맷팅
    candidates = format_candidates(merged_results, show=5)
    
    # BM25 분석 (로깅/학습데이터용, API 호출 없음)
    # src_lang에 맞는 BM25 retriever 선택
    bm25_retriever = cn_bm25_retriever if src_lang == "cn" else ko_bm25_retriever
    bm25_analysis = None
    if bm25_retriever:
        # BM25 검색용 쿼리 정규화 (브래킷 제거)
        query_for_bm25 = normalize_special_symbols(query)
        logger.debug("BM25 query normalized: %r -> %r", query, query_for_bm25)
        
        bm25_results = bm25_retriever.retrieve(query_for_bm25)
        # merged_results는 dict 형태로 전달
        bm25_analysis = _analyze_bm25_results_for_merged(normalized_query, bm25_results, merged_results)
        logger.info(
            "BM25 analysis: exact_rank=%s, hybrid_rank=%s, in_hybrid=%s",
            bm25_analysis.get("bm25_exact_rank"),
            bm25_analysis.get("bm25_top_rank_in_hybrid"),
            bm25_analysis.get("is_bm25_top_in_hybrid")
        )
    
    # 1단계의 Condition 4: exact_match가 없음 + BM25도 확신 없음 Check (아직 양방향 재시도 안했다면)
    if not should_retry_two_way:
        has_exact_match = any(
            normalize_for_exact_match(r.get("cn", "")) == normalized_query or
            normalize_for_exact_match(r.get("ko", "")) == normalized_query
            for r in merged_results
        )
        if not has_exact_match:
            bm25_exact_rank = bm25_analysis.get("bm25_exact_rank") if bm25_analysis else None
            
            # BM25에서도 정확 매칭 없으면 양방향 재시도
            if bm25_exact_rank is None:
                should_retry_two_way = True
                retry_reason = "no_exact_match_and_bm25_no_exact_match"
                
                logger.info("Retrying with two-way search (reason=%s)", retry_reason)
                merged_results = search(
                    query, cn_retriever, ko_retriever,
                    src_lang=src_lang, top_k=5,
                    fallback_to_two_way=True
                )
                logger.info("Two-way search returned %d results", len(merged_results))
    
    # ----- 상위 5개 후보 로깅 -----
    for idx, item in enumerate(merged_results[:5]):
        logger.info(
            "candidate[%d] score=%s, cn=%r, ko=%r",
            idx, item.get("score"), item.get("cn", ""), item.get("ko", "")
        )
    
    if not merged_results:
        logger.info("No results found. (merged_results is empty)")
        # 결과 없음 → LLM fallback 시도
        if use_llm_fallback:
            logger.info("No match, trying LLM fallback for query=%r, src_lang=%s", query, src_lang)
            translated = _try_llm_translate_term(query, src_lang, candidates=candidates)
            if translated:
                # LLM 번역 결과 후처리 (브래킷 제거)
                translation_cleaned = normalize_special_symbols(translated)
                
                return {
                    "query": query,
                    "src_lang": src_lang,
                    "translation": translation_cleaned,
                    "reason": TranslationReason.LLM_TERM_FALLBACK_NO_MATCH,
                    "candidates": candidates,
                    "bm25_analysis": bm25_analysis,
                    "passed_bm25_check": None,  # 결과 없어서 체크 불가
                    "passed_gap_check": None,
                }
        logger.info("LLM fallback is disabled or failed. Returning no_match.")
        
        return {
            "query": query,
            "src_lang": src_lang,
            "translation_ko": None,
            "reason": TranslationReason.NO_MATCH,
            "candidates": candidates,
            "bm25_analysis": bm25_analysis,
            "passed_bm25_check": None,
            "passed_gap_check": None,
        }
    
    # ================================
    # 1단계: 모든 candidates에서 정확 매칭 찾기 (공백 무시)
    # - 점수가 EXACT_MATCH_MIN_SCORE 이상인 경우만 신뢰
    # - normalized_query는 함수 시작 시 이미 계산됨
    # === 1단계 대응: 신뢰도 체크 준비 ===
    # bm25_exact_rank가 없으면 BM25에서 정확 매칭을 못 찾은 것 (신뢰도 낮음)
    bm25_exact_rank = bm25_analysis.get("bm25_exact_rank") if bm25_analysis else None
    
    # candidate_gap 계산 (1위-2위 점수 차이)
    # candidates가 1개면 경쟁자가 없으므로 gap 체크 skip (None으로 표시)
    candidate_gap = None
    if len(candidates) >= 2:
        candidate_gap = candidates[0]["score"] - candidates[1]["score"]
    
    # ================================
    for item in merged_results:
        r_cn = item.get("cn", "")
        r_ko = item.get("ko", "")
        r_score = item.get("score", 0)
        doc_type = item.get("matched_by", "")
        parent_cn = item.get("parent_cn", "")
        
        # 점수가 너무 낮으면 정확 매칭이라도 신뢰하지 않음 (오염된 데이터 방지)
        if r_score < EXACT_MATCH_MIN_SCORE:
            logger.debug("Skipping low-score candidate: cn=%r, ko=%r, score=%s (min=%s)", 
                        r_cn, r_ko, r_score, EXACT_MATCH_MIN_SCORE)
            continue
        
        # 정확 매칭 체크 (공백 무시)
        is_exact_cn = normalize_for_exact_match(r_cn) == normalized_query
        is_exact_ko = normalize_for_exact_match(r_ko) == normalized_query   # TODO: r_ko 정규화 여부 및 r_ko대신 r_cn의 metadata에서 ko를 추출할지 
        
        # 중국어 원문 → 한국어
        if src_lang == "cn" and is_exact_cn and r_ko:
            # === 1단계 대응: 신뢰도 체크 ===
            # 1. BM25에서도 정확 매칭이 있어야 신뢰
            # 2. candidate_gap이 너무 작으면 (동점) 확정 금지
            if bm25_exact_rank is None:
                logger.info("Exact match found but BM25 has no exact match, using as hint: cn=%r", r_cn)
                # BM25에서 정확 매칭 없으면 LLM fallback으로 (힌트로 전달)
                break
            
            # segment 매칭은 즉시 반환하지 않음 → LLM 힌트로 처리
            # (cn_segment 타입은 ko가 parent 전체 번역이므로 부정확)
            if doc_type == "cn_segment" and parent_cn:
                logger.info("Segment match found, skipping for LLM hint: cn=%r, parent_cn=%r", r_cn, parent_cn)
                continue  # 다음 candidate 확인
            
            # candidate_gap이 None이면 경쟁자 없음 → gap 체크 skip
            skip_gap_check = (doc_type != "cn_segment")
            if not skip_gap_check:
                if candidate_gap is not None and candidate_gap < CANDIDATE_GAP_THRESHOLD:
                    logger.info("Exact match found but candidate_gap too small (%.4f < %.4f), using as hint: cn=%r", 
                                candidate_gap, CANDIDATE_GAP_THRESHOLD, r_cn)
                    break
            
            # 일반 exact match
            logger.info("Exact match found: cn=%r, ko=%r, score=%s, bm25_rank=%s, gap=%s", 
                       r_cn, r_ko, r_score, bm25_exact_rank, 
                       f"{candidate_gap:.4f}" if candidate_gap is not None else "N/A")
            
            # 번역 결과 후처리 (브래킷 제거)
            translation_cleaned = normalize_special_symbols(r_ko)
            
            return {
                "query": query,
                "src_lang": src_lang,
                "translation": translation_cleaned,
                "reason": TranslationReason.EXACT_MATCH,
                "candidates": candidates,
                "bm25_analysis": bm25_analysis,
                "passed_bm25_check": True,
                "passed_gap_check": True,
            }
        
        # 한국어 원문 → 중국어
        if src_lang == "ko" and is_exact_ko and r_cn:
            # === 1단계 대응: 신뢰도 체크 (ko→cn도 동일하게 적용) ===
            if bm25_exact_rank is None:
                logger.info("Exact match found but BM25 has no exact match, using as hint: ko=%r", r_ko)
                break
            
            logger.info("Exact match found: cn=%r, ko=%r, score=%s", r_cn, r_ko, r_score)
            
            # 번역 결과 후처리 (브래킷 제거)
            translation_cleaned = normalize_special_symbols(r_cn)
            
            return {
                "query": query,
                "src_lang": src_lang,
                "translation": translation_cleaned,
                "reason": TranslationReason.EXACT_MATCH,
                "candidates": candidates,
                "bm25_analysis": bm25_analysis,
                "passed_bm25_check": True,
                "passed_gap_check": True,
            }
    # ================================
    # 정확 매칭 없음 - top 결과 정보 로깅
    top = merged_results[0]
    top_cn = top.get("cn", "")
    top_ko = top.get("ko", "")
    logger.info("No exact match in any candidate. Top result: cn=%r, ko=%r", top_cn, top_ko)
    
    # ================================
    # 1.5단계: 정확 매칭은 없지만 gap이 충분히 큰 경우 -> 1위 신뢰
    # - 1위가 2위보다 확실히 높은 점수면 LLM 없이 반환
    # - 단, BM25에서도 상위권에 있어야 함(추가 안전장치)
    # ================================
    if candidate_gap is not None and candidate_gap >= CANDIDATE_GAP_THRESHOLD:
        top_score = top.get("score", 0)
        
        # 추가 조건: 1위 점수가 최소 임계값 이상이어야 함.
        if top_score >= EXACT_MATCH_MIN_SCORE:
            # src_lang에 따라 번역 방향 결정            
            if src_lang == "cn" and top_ko:
                logger.info("No exact match but gap is large (%.4f >= %.4f), trusting top result: cn=%r -> ko=%r",
                            candidate_gap, CANDIDATE_GAP_THRESHOLD, top_cn, top_ko)
                
                translation_cleaned = normalize_special_symbols(top_ko)
                
                return {
                    "query": query,
                    "src_lang": src_lang,
                    "translation": translation_cleaned,
                    "reason": TranslationReason.TOP_CANDIDATE_HIGH_CONFIDENCE,
                    "candidates": candidates,
                    "bm25_analysis": bm25_analysis,
                    "passed_bm25_check": bm25_exact_rank is not None,
                    "passed_gap_check": True,
                }
            elif src_lang == "ko" and top_cn:
                logger.info("No exact match but gap is large, trusting top result: ko=%r -> cn=%r",
                            top_ko, top_cn)
                
                translation_cleaned = normalize_special_symbols(top_cn)
                
                return {
                    "query": query,
                    "src_lang": src_lang,
                    "translation": translation_cleaned,
                    "reason": TranslationReason.TOP_CANDIDATE_HIGH_CONFIDENCE,
                    "candidates": candidates,
                    "bm25_analysis": bm25_analysis,
                    "passed_bm25_check": bm25_exact_rank is not None,
                    "passed_gap_check": True,
                }
    
    # ================================
    # 2단계: 쿼리 세그먼트로 glossary 부분 매칭 시도 (힌트 전용)
    # - segment 매칭은 절대 full_match로 즉시 반환하지 않음
    # - 세그먼트 조합 번역은 문맥 오류 위험이 높으므로 LLM 힌트로만 활용
    # ================================
    glossary_hints = None
    if glossary_map and src_lang == "cn":
        segment_result = lookup_glossary_by_segments(
            query, glossary_map, src_lang, 
            candidates=candidates,
            bm25_retriever=bm25_retriever
        )
        if segment_result:
            matches = segment_result["matches"]
            unmatched = segment_result["unmatched_segments"]
            
            # is_segment=True 여부 확인 (ko가 parent 전체 번역인지)
            has_segment_ko = any(m.get("is_segment", False) for m in matches)
            similar_patterns = segment_result.get("similar_patterns", [])
            
            # 모든 segment 매칭은 LLM 힌트로만 사용 (full_match 금지)
            # - is_segment=True: ko가 parent 전체 번역이므로 부정확
            # - is_segment=False: 세그먼트 조합은 문맥 오류 위험
            glossary_hints = {
                "matches": matches,
                "unmatched": unmatched,
                "all_segments": segment_result["all_segments"],
                "has_segment_ko": has_segment_ko,
                "similar_patterns": similar_patterns,  # 유사 패턴 추가!
            }
            logger.info("Segment match found (hint only): matches=%r, unmatched=%r, has_segment_ko=%s, similar_patterns=%d", 
                       matches, unmatched, has_segment_ko, len(similar_patterns))
    
    # ================================
    # 3단계: '정확 매칭'이 없는 경우 LLM 번역 시도 + Glossary에 저장
    # ================================
    logger.info("No exact match for src_lang=%s (top_cn=%r, top_ko=%r), trying LLM fallback", src_lang, top_cn, top_ko)
    if use_llm_fallback:
        translated = _try_llm_translate_term(query, src_lang=src_lang, candidates=candidates, glossary_hints=glossary_hints)
        logger.info("LLM translation result: %r", translated)
        #  TODO: 무조건 자동 저장이 아닌 신뢰도에 따라 저장되는 로직으로 변경 예정
        if translated:
        #     # 새로운 용어를 Glossary에 저장 (자동 학습)
        #     cn_text = query if src_lang == "cn" else translated
        #     ko_text = translated if src_lang == "cn" else query
        #     add_user_entry_to_glossary(
        #         cn=cn_text,
        #         ko=ko_text,
        #         doc_type="user_term",
        #         src_lang=src_lang,
        #         reason="llm_term_fallback_no_exact_match"
        #     )
            
            # 신뢰도 체크 결과 (LLM fallback으로 온 이유 기록)
            passed_bm25 = bm25_exact_rank is not None
            # candidate_gap이 None이면 경쟁자 없음 → True로 처리
            passed_gap = candidate_gap is None or candidate_gap >= CANDIDATE_GAP_THRESHOLD
            
            # LLM 번역 결과도 후처리 (브래킷 제거)
            translation_cleaned = normalize_special_symbols(translated)
            
            return {
                "query": query,
                "src_lang": src_lang,
                "translation": translation_cleaned,
                "reason": TranslationReason.LLM_TERM_FALLBACK_NO_EXACT_MATCH,
                "candidates": candidates,
                "bm25_analysis": bm25_analysis,
                "glossary_hints": glossary_hints,
                "passed_bm25_check": passed_bm25,
                "passed_gap_check": passed_gap,
            }
    
    # ================================
    # 3단계: LLM도 번역 불가 시, 추천 후보만 돌려주고 번역 없음
    # ================================
    passed_bm25 = bm25_exact_rank is not None
    # candidate_gap이 None이면 경쟁자 없음 → True로 처리
    passed_gap = candidate_gap is None or candidate_gap >= CANDIDATE_GAP_THRESHOLD
    
    return {
        "query": query,
        "src_lang": src_lang,
        "translation_ko": None,
        "reason": TranslationReason.NO_KO_AND_LLM_FAILED,
        "candidates": candidates,
        "bm25_analysis": bm25_analysis,
        "glossary_hints": glossary_hints,
        "passed_bm25_check": passed_bm25,
        "passed_gap_check": passed_gap,
    }
    
    
def translate_sentence_with_glossary(
    sentence: str,
    cn_retriever: "QueryFusionRetriever",
    ko_retriever: "QueryFusionRetriever",
    src_lang: str,
    use_llm: bool = True,
    use_tm: bool = True,
    tm_retriever: "QueryFusionRetriever" = None
) -> dict:
    """
    용어 사전 + Translation Memory를 활용한 문장 번역
    
    처리 순서:
    1. [TM] Translation Memory 검색 → 완전 일치 시 즉시 반환
    2. [TM] 유사 문장 발견 시 참고용으로 저장
    3. 문장에서 용어 후보 추출
    4. VectorDB에서 각 용어 검색하여 glossary 구성 (src_lang 기반 선택적)
    5. LLM에 원문 + glossary + (유사 문장 참고) 제공하여 번역
    
    Args:
        sentence: 번역할 문장
        cn_retriever: CN 컬렉션 QueryFusionRetriever (용어 검색용)
        ko_retriever: KO 컬렉션 QueryFusionRetriever (용어 검색용)
        src_lang: 원본 언어 ("cn" 또는 "ko")
        use_llm: LLM 사용 여부
        use_tm: Translation Memory 사용 여부
        tm_retriever: TM 전용 retriever (num_queries=1, LLM 호출 없음)
    
    Returns:
        {
            "query": 원본 문장,
            "src_lang": 원본 언어,
            "translation": 번역 결과,
            "glossary": 사용된 용어 사전,
            "tm_match": TM 매칭 정보 (있으면),
            "reason": 번역 방식
        }
    """
    logger.info("translate_sentence_with_glossary called, sentence=%r, src_lang=%s", sentence, src_lang)
    
    tm_match = None
    
    # src_lang에 따라 적절한 retriever 선택 (TM과 glossary에서 공용)
    hybrid = cn_retriever if src_lang == "cn" else ko_retriever
    
    # ===== 0단계: Translation Memory 검색 =====
    # tm_retriever가 없으면 hybrid 사용 (하위 호환)
    if use_tm:
        tm_searcher = tm_retriever if tm_retriever else hybrid
        tm_match = search_translation_memory(sentence, tm_searcher, src_lang)
        
        # 완전 일치 → 즉시 반환 (LLM 호출 없이 reuse)
        if tm_match and tm_match["match_type"] == "exact":
            logger.info("[TM] Exact match found, reusing translation")
            return {
                "query": sentence,
                "src_lang": src_lang,
                "translation": tm_match["translation"],
                "glossary": {},
                "tm_match": tm_match,
                "reason": TranslationReason.TM_EXACT_REUSE
            }
        
        # 유사 문장 → 나중에 LLM 프롬프트에서 참고
        if tm_match and tm_match["match_type"] == "similar":
            logger.info("[TM] Similar match found, will use as reference")
    
    # ===== 1단계: 용어 후보 추출 =====
    terms = extract_terms_from_sentence(sentence)
    logger.info("Extracted %d term candidates", len(terms))
    
    # ===== 2단계: VectorDB에서 glossary 구성 (src_lang 기반 선택적 검색) =====
    glossary = lookup_glossary(terms, hybrid, src_lang, max_hybrid_calls=3)
    logger.info("Glossary (VectorDB): %d matches - %s", len(glossary), glossary)
    
    # ===== 2.5단계: 역방향 lookup (한국어 → 중국어) =====
    # 한국어 문장인 경우, Glossary ko가 문장에 포함되어 있는지 추가 체크
    if src_lang == "ko":
        glossary_map_ko = _get_glossary_map_ko()
        reverse_matches = lookup_glossary_in_sentence(sentence, glossary_map_ko)
        
        # 병합 (기존 glossary에 없는 것만 추가)
        for ko_term, cn_trans in reverse_matches.items():
            if ko_term not in glossary:
                glossary[ko_term] = cn_trans
                logger.info("Added from reverse lookup: %r → %r", ko_term, cn_trans)
    
    logger.info("Glossary final: %d matches - %s", len(glossary), glossary)
    
    # ===== 3단계: LLM 번역 =====
    if not use_llm or not (hasattr(Settings, "llm") and Settings.llm):
        return {
            "query": sentence,
            "src_lang": src_lang,
            "translation": None,
            "glossary": glossary,
            "tm_match": tm_match,
            "reason": TranslationReason.LLM_NOT_AVAILABLE
        }
    
    # ===== 프롬프트 구성 =====
    # 1. 용어 사전 텍스트
    glossary_text = ""
    if glossary:
        glossary_text = "\n".join([f"- {k} → {v}" for k, v in glossary.items()])
    
    # 2. 유사 문장 참고 텍스트 (TM partial reuse)
    tm_reference_text = ""
    if tm_match and tm_match["match_type"] == "similar":
        tm_reference_text = f"""
**참고: 유사한 문장의 기존 번역 (유사도 {tm_match['similarity']}%)**
- 원문: {tm_match['source']}
- 번역: {tm_match['translation']}
→ 위 번역을 참고하되, 현재 문장에 맞게 조정하세요."""
    
    # 3. 프롬프트 조립
    if src_lang == "cn":
        target_lang = "한국어"
        source_lang = "중국어"
    else:
        target_lang = "중국어(간체)"
        source_lang = "한국어"
    
    prompt_parts = [f"다음 {source_lang} 문장을 {target_lang}로 번역해주세요."]
    
    # 용어 사전 추가
    if glossary_text:
        prompt_parts.append(f"""
                            **반드시 아래 용어 사전의 번역을 사용하세요:**
                            {glossary_text}
                            """)
    
    # TM 참고 문장 추가
    if tm_reference_text:
        prompt_parts.append(tm_reference_text)
    
    # 규칙 추가
    prompt_parts.append("""
                        **규칙:**
                        - 용어 사전에 있는 단어는 반드시 해당 번역을 사용하세요.
                        - 게임 맥락에 자연스럽게 어울리도록 번역하세요.
- 번역 결과만 출력하세요.""")
    
    # 원문 추가
    prompt_parts.append(f"""
원문: {sentence}
번역: """)
    
    prompt = "\n".join(prompt_parts)
    
    try:
        resp = Settings.llm.complete(prompt)
        translation = str(resp).strip()
        
        # reason 결정: TM 참고 여부 + Glossary 사용 여부
        if tm_match and tm_match["match_type"] == "similar":
            reason = TranslationReason.TM_PARTIAL_REUSE  # 유사 문장 참고하여 번역
        elif glossary:
            reason = TranslationReason.GLOSSARY_AWARE_TRANSLATION  # 용어 사전 참고하여 번역
        else:
            reason = TranslationReason.LLM_TRANSLATION  # 순수 LLM 번역
        
        # TODO: 무조건 자동 저장이 아닌 신뢰도에 따라 저장되는 로직으로 변경 예정
        # 번역 결과를 Glossary에 저장 (Translation Memory로 활용)
        # cn_text = sentence if src_lang == "cn" else translation
        # ko_text = translation if src_lang == "cn" else sentence
        # add_user_entry_to_glossary(
        #     cn=cn_text,
        #     ko=ko_text,
        #     doc_type="user_sentence",
        #     src_lang=src_lang,
        #     reason=reason
        # )
        
        return {
            "query": sentence,
            "src_lang": src_lang,
            "translation": translation,
            "glossary": glossary,
            "tm_match": tm_match,  # TM 매칭 정보 포함
            "reason": reason
        }
    
    except Exception as e:
        logger.error("Glossary-aware translation failed: %s", e)
        return {
            "query": sentence,
            "src_lang": src_lang,
            "translation": None,
            "glossary": glossary,
            "tm_match": tm_match,
            "reason": TranslationReason.LLM_FAILED
        }


# ===============================================
# LLM 번역 Helper Function (term 기준)
# ===============================================
def _try_llm_translate_term(query: str, src_lang: str, candidates: list[dict] = None, glossary_hints: dict = None) -> str | None:
    """
    LLM을 사용해 단어 번역 시도. 실패 시 None 반환.
    
    Args:
        query: 번역할 용어
        src_lang: 원본 언어 ("cn" 또는 "ko")
        candidates: VectorDB 검색 결과 (후보 참고용)
        glossary_hints: glossary에서 찾은 부분 매칭 정보
            {
                "matches": [{"cn": "全联堂", "ko": "전연당", "is_segment": True}, ...],
                "unmatched": ["场景", "赠送"],
                "all_segments": ["全联堂", "场景", "赠送"]
            }
    """
    if not (hasattr(Settings, "llm") and Settings.llm):
        return None
    
    # 후보 정보를 참고 텍스트로 변환
    reference_text = ""
    if candidates:
        ref_lines = []
        for c in candidates[:5]:  # 상위 5개만
            cn = c.get("cn", "")
            ko = c.get("ko", "")
            if cn and ko:
                ref_lines.append(f"  - {cn} → {ko}")
        if ref_lines:
            reference_text = "\n".join(ref_lines)
    
    # glossary 힌트를 상세한 번역 규칙으로 변환
    glossary_section = ""
    if glossary_hints and isinstance(glossary_hints, dict):
        matches = glossary_hints.get("matches", [])
        unmatched = glossary_hints.get("unmatched", [])
        all_segments = glossary_hints.get("all_segments", [])
        similar_patterns = glossary_hints.get("similar_patterns", [])
        
        # 정확한 번역 수집 (exact_ko 우선 사용)
        exact_matches = []
        reference_terms = []
        for m in matches:
            cn = m.get("cn", "")
            ko = m.get("ko", "")
            exact_ko = m.get("exact_ko")  # 원본 cn의 정확한 번역
            is_seg = m.get("is_segment", False)
            
            if cn:
                if exact_ko:
                    # exact_ko가 있으면 정확한 번역으로 사용 (is_segment 여부와 무관)
                    exact_matches.append(f"  - '{cn}' → '{exact_ko}'")
                elif not is_seg and ko:
                    # is_segment=False이고 ko가 있으면 필수 규칙
                    exact_matches.append(f"  - '{cn}' → '{ko}'")
                else:
                    # exact_ko도 없고 is_segment=True면 참고 용어로만 표시
                    reference_terms.append(cn)
        
        # 필수 번역 규칙 (정확한 ko가 있는 것만)
        if exact_matches:
            glossary_section += f"""
*필수 용어 번역 규칙 (반드시 아래 번역을 사용하세요)*
{chr(10).join(exact_matches)}
"""
        
        # 참고 용어 (glossary에 있지만 정확한 ko 추출 불가)
        if reference_terms:
            ref_text = ", ".join([f"'{t}'" for t in reference_terms])
            glossary_section += f"""
*참고 용어 (glossary에 존재하는 단어, 문맥에 맞게 번역)*
  - {ref_text}
"""
        
        # 유사 패턴 참조 (같은 suffix를 가진 기존 번역)
        if similar_patterns:
            pattern_lines = []
            for p in similar_patterns[:3]:  # 최대 3개
                pattern_lines.append(f"  - '{p['pattern']}' → '{p['translation']}'")
            glossary_section += f"""
*유사 패턴 참조 (동일한 패턴의 기존 번역, 일관된 번역 규칙 적용)*
{chr(10).join(pattern_lines)}
"""
        
        # 번역 필요 부분
        if unmatched:
            unmatched_text = ", ".join([f"'{u}'" for u in unmatched if len(u) > 1])
            if unmatched_text:
                glossary_section += f"""
*번역 필요 부분 (glossary에 없음, 자연스럽게 번역해주세요)*
  - {unmatched_text}
"""
        
        # 세그먼트 순서
        if all_segments and len(all_segments) > 1:
            segment_order = " → ".join(all_segments)
            glossary_section += f"""
*세그먼트 순서 (번역 시 순서 유지)*
  {segment_order}
"""
    
    if src_lang == "cn":
        prompt = f"""
다음 중국어의 게임 용어를 자연스러운 한국어 게임용어로 번역해주세요. 

*중요 규칙*
- 번역 결과만 출력하세요.
- 설명, 예시, 추가 문장, 따옴표는 출력하지 마세요.
- 한 줄로만 출력하세요.
- 아래 참고 번역에서 유사한 패턴을 찾아 일관된 번역을 해주세요.
{glossary_section}
*참고 번역 (용어사전에서 검색된 유사 표현)*
{reference_text if reference_text else "  (없음)"}

중국어: {query}

한국어: """.strip()
        
    elif src_lang == "ko":
        prompt = f"""
다음 한국어의 게임 용어를 자연스럽고 간결한 중국어(간체) 게임용어로 번역해주세요. 

*중요 규칙*
- 번역 결과만 출력하세요.
- 설명, 예시, 추가 문장, 따옴표는 출력하지 마세요.
- 한 줄로만 출력하세요.
- 아래 참고 번역에서 유사한 패턴을 찾아 일관된 번역을 해주세요.

*참고 번역 (용어사전에서 검색된 유사 표현)*
{reference_text if reference_text else "  (없음)"}

한국어: {query}
중국어: """.strip()
        
    else:
        prompt = f"""
다음 게임 용어를 보고, 원문 언어를 추론한 후 반대 언어(중국어<->한국어)로 번역해주세요. 

*중요 규칙*
- 번역 결과만 출력하세요.
- 설명, 예시, 추가 문장, 따옴표는 출력하지 마세요.
- 한 줄로만 출력하세요.

게임 용어: {query}

번역: """.strip()
    
    try:
        resp = Settings.llm.complete(prompt)
        return str(resp)
    except Exception as e:
        logger.error("LLM term translate failed: %s", e)
        return None

# ===============================================
# Translation Memory (번역 메모리)
# ===============================================    

# TM 매칭 임계값
TM_EXACT_THRESHOLD = 0.032    # 이 이상이면 완전 일치로 간주 (reuse)
TM_SIMILAR_THRESHOLD = 0.028  # 이 이상이면 유사 문장으로 간주 (partial reuse)

def search_translation_memory(
    sentence: str,
    tm_retriever: "QueryFusionRetriever",
    src_lang: str,
    top_k: int = 3
) -> dict | None:
    """
    Translation Memory 검색 - 이전에 번역된 유사 문장 찾기
    
    Args:
        sentence: 번역할 문장
        tm_retriever: TM 전용 retriever (num_queries=1, LLM 호출 없음)
        src_lang: 원본 언어 ("cn" 또는 "ko")
        top_k: 검색할 최대 결과 수
    
    Returns:
        매칭된 경우:
        {
            "match_type": "exact" | "similar",  # 매칭 유형
            "source": str,       # 원문 (DB에 저장된)
            "translation": str,  # 번역문 (DB에 저장된)
            "score": float,      # 유사도 점수
            "similarity": float  # 백분율 유사도 (0~100)
        }
        매칭 없으면: None
    """
    # TM 전용 retriever로 유사 문장 조회 (LLM 호출 없음, Embedding만 사용)
    results = tm_retriever.retrieve(sentence)[:top_k]
    
    # 결과 없으면 None 반환
    if not results:
        logger.debug("[TM] No results for: %r", sentence)
        return None
    
    # 최상위 결과 분석
    top = results[0]
    score = top.score or 0.0
    meta = top.node.metadata
    
    # 메타데이터에서 cn, ko 추출
    top_cn = meta.get("cn", "")
    top_ko = meta.get("ko", "")
    
    # 번역쌍이 없으면 TM으로 사용 불가
    if not top_cn or not top_ko:
        logger.debug("[TM] No translation pair in result")
        return None
    
    # 원본과 번역 결정 (src_lang 기준)
    if src_lang == "cn":
        source = top_cn       # DB의 중국어
        translation = top_ko  # DB의 한국어
    else:
        source = top_ko       # DB의 한국어
        translation = top_cn  # DB의 중국어
    
    # 완전 일치 판정 (텍스트 동일 또는 높은 점수)
    is_text_exact = (sentence.strip() == source.strip())
    is_score_exact = (score >= TM_EXACT_THRESHOLD)
    
    if is_text_exact or is_score_exact:
        # 완전 일치 → 그대로 reuse
        similarity = min(100.0, score * 3000)  # 대략적 백분율 변환
        logger.info("[TM] EXACT match: %r → %r (score=%.4f)", sentence, translation, score)
        return {
            "match_type": "exact",
            "source": source,
            "translation": translation,
            "score": score,
            "similarity": round(similarity, 1)
        }
    
    # 유사 문장 판정
    if score >= TM_SIMILAR_THRESHOLD:
        # 유사 문장 → partial reuse (LLM 참고용)
        similarity = min(100.0, score * 3000)
        logger.info("[TM] SIMILAR match: %r ≈ %r (score=%.4f)", sentence, source, score)
        return {
            "match_type": "similar",
            "source": source,
            "translation": translation,
            "score": score,
            "similarity": round(similarity, 1)
        }
    
    # 임계값 미달 → 매칭 없음
    logger.debug("[TM] Below threshold: score=%.4f < %.4f", score, TM_SIMILAR_THRESHOLD)
    return None

def _analyze_bm25_results_for_merged(normalized_query: str, bm25_results, merged_results: list[dict]) -> dict:
    """
    BM25 결과를 분석하여 로깅용 정보 생성 (merged_results용)
    - merged_results와 비교하여 Vector 기여도 추론
    
    Args:
        normalized_query: 정규화된 검색 쿼리
        bm25_results: BM25 retriever 결과 (NodeWithScore)
        merged_results: search()로 병합된 결과 (list[dict])
    
    Returns:
        {
            "bm25_top_score": BM25 최상위 점수,
            "bm25_exact_rank": BM25에서 정확매칭 순위 (없으면 None),
            "bm25_exact_match": 정확매칭 내용 {cn, ko},
            "is_bm25_top_in_hybrid": BM25 top이 merged에도 있는지,
            "bm25_top_rank_in_hybrid": merged에서의 BM25 top 순위,
        }
    """
    
    # BM25에서 정확 매칭 찾기
    bm25_exact_match = None
    bm25_top_score = None
    bm25_exact_rank = None
    bm25_top_cn = None
    
    for rank, r in enumerate(bm25_results):
        md = r.node.metadata or {}
        r_cn = md.get("cn", "")
        r_ko = md.get("ko", "")
        
        if rank == 0:
            bm25_top_score = getattr(r, "score", None)
            bm25_top_cn = r_cn
        
        is_exact = (
            normalize_for_exact_match(r_cn) == normalized_query or
            normalize_for_exact_match(r_ko) == normalized_query
        )
        
        if is_exact and bm25_exact_rank is None:
            bm25_exact_rank = rank + 1
            bm25_exact_match = {"cn": r_cn, "ko": r_ko}
    
    # merged_results에서 BM25 top 항목 찾기 (Vector 기여 추론)
    merged_cn_list = [item.get("cn", "") for item in merged_results]
    
    is_bm25_top_in_hybrid = bm25_top_cn in merged_cn_list if bm25_top_cn else False
    bm25_top_rank_in_hybrid = (merged_cn_list.index(bm25_top_cn) + 1) if is_bm25_top_in_hybrid else None
    
    return {
        "bm25_top_score": bm25_top_score,       # BM25 top 점수
        "bm25_exact_rank": bm25_exact_rank,     # BM25에서 정확 매칭 순위
        "bm25_exact_match": bm25_exact_match,   # BM25 정확 매칭 내용 {cn, ko}
        "is_bm25_top_in_hybrid": is_bm25_top_in_hybrid,       # BM25 top이 merged에 있는지
        "bm25_top_rank_in_hybrid": bm25_top_rank_in_hybrid,   # merged에서의 BM25 top 순위
    }
