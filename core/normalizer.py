# core/normalizer.py
# Text 정규화, 토큰화 유틸리티

import re
import jieba
from functools import lru_cache

_CJK_SEPARATORS = "·•・∙⋅-—_"
_SEP_PATTERNS = re.compile(f"[{re.escape(_CJK_SEPARATORS)}]+")


def to_embedding_normalize_for_search(text: str) -> str:
    """
    임베딩 검색용 정규화: 공백 제거 + 구분자 제거
    """
    if not text:
        return ""
    unblanked = _SEP_PATTERNS.sub(" ", text.strip())    # 구분자를 제거하고 공백으로 치환
    unblanked = re.sub(r"\s+", " ", unblanked).strip()  # 연속된 공백을 하나의 공백으로 치환
    return unblanked


def split_text_segments(text: str) -> list[str]:
    """
    중국어 문장 부분 매칭용: 복합 텍스트를 세그먼트로 분해(임베딩 검색용)
    - 먼저 구분자로 split
    - 각 부분을 jieba로 추가 분해 (중국어 단어 경계 인식)
    - 1글자인 경우 noise일 수 있어 제거(길이 2 이상)
    
    예: "期间限定" → ["期间", "限定"]
        "攀登者·综合速配" → ["攀登者", "综合", "速配"]
    """
    normalized = to_embedding_normalize_for_search(text)
    if not normalized:
        return []
    
    # 1단계: 구분자로 split
    parts = [p for p in normalized.split(" ") if p]
    
    # 2단계: 각 part를 jieba로 추가 분해
    all_segments = []
    for part in parts:
        jieba_segments = list(jieba.cut(part))
        all_segments.extend([s for s in jieba_segments if len(s) >= 2])
    
    return all_segments


def split_by_separator(text: str) -> list[str]:
    """
    구분자 기준으로만 텍스트 분리 (CN/KO 매핑용)
    - jieba 없이 구분자(·•・∙⋅-—_)로만 분리
    - CN과 KO의 segment 개수를 맞추기 위해 사용
    
    예: "攀登者·综合速配" → ["攀登者", "综合速配"]
        "등반가·종합 패키지" → ["등반가", "종합 패키지"]
    """
    if not text:
        return []
    
    # 구분자를 공백으로 치환 후 분리
    normalized = _SEP_PATTERNS.sub(" ", text.strip())
    parts = [p.strip() for p in normalized.split(" ") if p.strip()]
    
    return parts


def split_segments_paired(cn: str, ko: str) -> list[tuple[str, str]]:
    """
    CN과 KO를 구분자 기준으로 분리하고 쌍으로 매핑
    - 개수가 같으면 순서대로 매핑
    - 개수가 다르면 빈 리스트 반환 (매핑 불가)
    
    예: 
        cn="攀登者·综合速配", ko="등반가·종합 패키지"
        → [("攀登者", "등반가"), ("综合速配", "종합 패키지")]
    
    Returns:
        [(cn_segment, ko_segment), ...] 또는 [] (매핑 불가 시)
    """
    cn_parts = split_by_separator(cn)
    ko_parts = split_by_separator(ko)
    
    # 개수가 다르면 매핑 불가
    if len(cn_parts) != len(ko_parts):
        return []
    
    # 1개면 segment 아님 (원본 전체)
    if len(cn_parts) <= 1:
        return []
    
    return list(zip(cn_parts, ko_parts))


# -----------------------------
# lru_cache: least recently used cache
# - 캐시 크기 제한: 2048
# - 캐시 동작 방식: 가장 오래된 항목부터 제거
# -----------------------------
@lru_cache(maxsize=2048)
def normalize_for_exact_match(text: str) -> str:
    """
    정확 매칭 비교용(강하게 비교): 공백 제거 + 구분자 제거
    - 공백 제거 + 구분자 제거
    - lru_cache로 동일 입력 재계산 방지
    """
    if not text:
        return ""
    
    return to_embedding_normalize_for_search(text).replace(" ", "")


def chinese_tokenizer(text: str) -> list[str]:
    """
    중국어 분사 토크나이저 (jieba 사용)
    - BM25가 중국어 단어 경계를 인식하도록 함
    - 예: "全联堂场景赠送" → ["全联堂", "场景", "赠送"]
    """
    return list(jieba.cut(text))
