


def calculate_confidence(top_score, bm25_exact_rank, candidate_gap, has_segment):
    
    # None 처리
    top_score = float(top_score or 0.0)
    candidate_gap = float(candidate_gap or 0.0)
    
    # confidence 초기화
    confidence = 0.0
    
    # top_score 기반 confidence 계산
    if top_score >= 0.05:
        confidence += 0.45
    elif top_score >= 0.045:
        confidence += 0.25
    elif top_score >= 0.03:
        confidence += 0.10
    
    # BM25 기반 confidence를 강하게 계산
    if bm25_exact_rank == 1:
        confidence += 0.35
    elif bm25_exact_rank is not None:
        confidence += 0.05
    
    # candidate_gap 기반 confidence 계산
    if candidate_gap >= 0.02 or candidate_gap is None:
        confidence += 0.15
    elif candidate_gap >= 0.01:
        confidence += 0.05
    
    # segment 처리: 무조건 패널티가 아닌 "불확실성"을 가중치로 적용
    # segment인데, bm25_exact_rank=1이면 패널티 거의 없음(keyword로도 확실하니까)
    # segment인데, top_score도 낮으면 불확실성 가중치 높게
    if has_segment:
        if bm25_exact_rank == 1:
            confidence -= 0.05
        elif top_score < 0.04:
            confidence -= 0.2
        else:
            confidence -= 0.1
    
    # confidence 결과를 0~1로 고정하는 sigmoid 함수 적용
    
    pass