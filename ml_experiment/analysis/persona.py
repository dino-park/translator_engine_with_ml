import pandas as pd


DEFINED_PERSONAS = {
    
}


def analyze_persona(row: pd.Series) -> str:
    """
    cluster 분석 결과를 기반으로 2~3개 feature 조합으로 persona 분석/결정
    """
    
    is_exact = row["is_exact_match"] == 1
    is_bm25 = row["is_bm25_match"] == 1
    is_llm = row["is_llm_fallback"] == 1
    is_segment = row["is_top_segment"] == 1
    is_segment_exact = row["is_segment_exact_match"] == 1
    bm25_rank = row.get("bm25_exact_rank", None)
    doc_type = row.get("top_doc_type", "") or ""
    
    top_score = float(row.get("top_score") or 0.0)
    gap = float(row.get("candidate_gap") or 0.0)
    has_candidates = len(row.get("candidates", [])) > 0
    
    
    # ================================
    # 1. Vector 검색 없이 처리된 Case (candidates가 없는 경우)
    # ================================
    if not has_candidates:
        return "sentence_llm" if is_llm else "sentence_tm"
        
    # ================================
    # 2. 정확 매칭이고 LLM fallback 없는 case (is_exact=1, is_llm=0)
    # ================================
    if is_exact and not is_llm:
        # 구조상 segment는 ko 위험이 있어 보수적으로 "검토"를 위한 분류
        if doc_type == "cn_segment" or is_segment_exact:
            return "exact_segment"
        if bm25_rank == 1 or is_bm25:
            return "exact_bm25"
        return "exact_vector"
    
    # ================================
    # 3. 부분 매칭(not exact), LLM 보완 case
    #    - 후보는 있었으나, 정확 매칭이 아니어서 LLM으로 보완
    # ================================
    if (not is_exact) and is_llm:
        if doc_type == "cn_segment" or is_segment:
            return "partial_segment_llm"            # Segment 힌트 + LLM
        elif bm25_rank == 1 or is_bm25:
            return "partial_bm25_llm"                # BM25 힌트 + LLM
        else:
            return "partial_vector_llm"              # Vector 힌트 + LLM
    
    # ================================
    # 4. exact 매칭도 아니고, LLM도 미사용 case
    # ================================
    if (not is_exact) and (not is_llm):
        return "no_match_other"
    
    # ================================
    # 5. 비정상 case에 대한 분석을 위한 추적(is_exact=1 and is_llm=1)
    #    - 정확 매칭인데 llm을 사용?
    # ================================
    if is_exact and is_llm:
        return "exact_but_llm"
    
    return "unknown"
    