"""
전처리 함수들 (Preprocessor)
- 쿼리 전처리
- 의도 분석
- 용어 추출
"""

import re
import logging

from utils import has_cjk, has_hangul

logger = logging.getLogger("preprocessor")


# ===============================================
# 정규식 패턴
# ===============================================

# 제거할 패턴들 (번역 요청 문구)
_REMOVE_PATTERNS = re.compile(
    r"[을를이가]?\s*"  # 조사 (선택)
    r"번역\s*"         # "번역" + 공백 (선택)
    r"(해\s*줘|해\s*주세요|해\s*줄래|좀|해봐|알려\s*줘|뭐야|뭔가요?|이?야)"  # 동사/어미
    r"[~?!.]*",        # 후행 기호
    re.IGNORECASE
)


# ===============================================
# 쿼리 전처리
# ===============================================

def preprocess_query(query: str) -> str:
    """
    쿼리 전처리: 불필요한 단어 제거 & 한자 추출
    
    1. 정규식으로 "번역해줘", "를 번역해주세요" 등 부탁 문구 제거
    2. 중국어(한자)가 있으면 가장 긴 한자 문자열 반환
    3. 한자 없으면 정리된 텍스트 반환
    
    Args:
        query: 원본 쿼리
    
    Returns:
        정리된 쿼리 문자열
    
    예시:
        "超现象管理局을 번역해줘" → "超现象管理局"
        "월상석 가루 번역해주세요" → "월상석 가루"
    """
    if not query:
        return ""
    
    text = query.strip()
    
    # 1. 부탁 문구 제거
    text = _REMOVE_PATTERNS.sub("", text).strip()
    
    # 2. 한자 추출 (구분자 포함하여 가장 긴 것)
    # 구분자(·•・∙⋅-_:;)도 포함하여 연속된 한자 문자열로 추출
    # 전각 콜론 \uff1a (：), 전각 세미콜론 \uff1b (；) 추가
    # 전각 괄호 \u3010【 \u3011】, 숫자 0-9, + 기호 추가
    hanja_matches = re.findall(r"[\u4e00-\u9fff·•・∙⋅\-_\s:;\uff1a\uff1b\u3010\u30110-9+]+", text)
    
    # 실제 한자가 포함된 매칭만 필터링 (공백/구분자만 있는 경우 제외)
    hanja_matches = [m for m in hanja_matches if re.search(r"[\u4e00-\u9fff]", m)]
    
    if hanja_matches:
        # 구분자로 끝나는 경우 제거
        result = max(hanja_matches, key=len)
        result = re.sub(r"[·•・∙⋅\-_]+$", "", result)  # 끝의 구분자 제거
        return result
    
    # 3. 한자 없으면 정리된 텍스트 반환
    return text


def normalize_query_with_llm(raw_query: str, llm) -> str:
    """
    LLM을 사용해 쿼리 정규화: 불필요한 단어 제거 & 핵심 용어 추출
    
    Args:
        raw_query: 원본 쿼리
        llm: LlamaIndex LLM 인스턴스
    
    Returns:
        정규화된 쿼리 문자열
    """
    if not llm:
        return raw_query
    
    prompt = f"""
    사용자가 게임 용어의 번역을 요청했습니다.
    
    아래 문장에서 '번역이 필요한 핵심 용어'만 추출해 주세요.
    
    **중요 규칙**
    - 게임 용어는 여러 단어로 구성될 수 있습니다. 공백으로 구분된 단어들도 하나의 용어일 수 있으니 함부로 제거하지 마세요.
    - "픽업", "UP", "단독", "한정", "이벤트", "패키지" 등은 게임 용어의 일부입니다. 절대 제거하지 마세요.
    - 오직 "번역해줘", "알려줘", "뭐야", "뭔가요", "입니다" 같은 명백한 부탁/설명 문구만 제거하세요.
    - 중국어(한자)가 있다면 그 용어만 출력하세요.
    - 잘 모르겠으면 입력을 그대로 출력하세요.
    
    사용자 입력:
    {raw_query}
    
    출력 형식:
    - 핵심 용어만 한 줄로 출력하세요. 설명 없이 용어만 출력하세요.
    - 예: "월광석을 번역해줘" -> "월광석"
    - 예: "부지춘 단독UP 픽업이 뭐야" -> "부지춘 단독UP 픽업"
    - 예: "한정 이벤트 패키지 번역" -> "한정 이벤트 패키지"
    """
    
    try:
        resp = llm.complete(prompt)
        text = str(resp).strip()
        return text
    except Exception as e:
        logger.error("normalize_query_with_llm failed: %s", e)
        return raw_query


def preprocess_query_smart(raw: str, llm=None) -> str:
    """
    스마트 전처리: 규칙 기반 → (필요시) LLM 보정
    
    Args:
        raw: 원본 쿼리
        llm: LlamaIndex LLM 인스턴스 (None이면 LLM 보정 건너뜀)
    
    Returns:
        전처리된 쿼리 문자열
    """
    q = preprocess_query(raw)
    logger.debug("preprocess_query: %r → %r", raw, q)
    
    # 한자(중국어)가 이미 있다면, 그대로 반환
    if has_cjk(q):
        return q
    
    # 15자 이하의 단어이고 공백이 없다면, 그대로 반환 (LLM 보정 필요 없음)
    if len(q) <= 15 and " " not in q:
        return q
    
    # LLM 보정이 필요한 패턴 정의
    is_long_sentence = len(q) > 15
    is_multi_word = " " in q and len(q) > 10
    has_question_keyword = any(kw in raw for kw in ("뭐야", "알려", "뭔가", "설명", "뜻이", "의미"))
    needs_llm = is_long_sentence or is_multi_word or has_question_keyword
    
    # LLM 보정 실행
    if needs_llm and llm:
        q_llm = normalize_query_with_llm(raw, llm)
        logger.debug("normalize_query_with_llm: %r → %r", raw, q_llm)
        return q_llm.strip() if q_llm else q
    
    return q


# ===============================================
# 의도 분석
# ===============================================

def detect_intent(raw: str) -> dict:
    """
    입력 텍스트의 의도 분석: term(용어) vs sentence(문장)
    
    판단 기준:
    - 중국어: 12자 이상 또는 문장 부호 포함 시 sentence
    - 한국어: 12자 이상, 공백 포함, 또는 특정 어미로 끝나면 sentence
    
    Args:
        raw: 원본 텍스트
    
    Returns:
        {"mode": "term" | "sentence", "src_lang": "cn" | "ko" | "unknown"}
    """
    text = raw.strip()
    cjk = has_cjk(text)
    hangul = has_hangul(text)
    
    # 중국어 문장 부호
    cn_sentence_markers = ("。", "？", "！", "，", "、")
    
    if cjk:
        # 중국어: 길이 또는 문장 부호로 sentence 판단
        is_sentence = len(text) > 12 or any(m in text for m in cn_sentence_markers)
        return {
            "mode": "sentence" if is_sentence else "term",
            "src_lang": "cn"
        }
    
    if hangul:
        # 개선2 버전 코드: 공백만으로 판단하지 않고, 글자수와 단어수로 판단
        word_count = len(text.split())
        is_sentence = (
            len(text) > 15 or # 글자수 15자 이상
            word_count > 4 or # 단어수 4개 이상
            text.endswith(("다", "요", "습니다.", ".")) # 문장 끝나는 패턴
        )
        
        # 개선1 버전 코드: 글자수 or 글자수and공백 or 문장 끝나는 패턴
        # if len(text) > 12 or (len(text) > 8 and " " in text) or text.endswith(("다", "요", "습니다.", ".")):
        
        # 기존코드: 글자수 or 공백으로 구분 or 문장 끝나는 패턴
        # if len(text) > 12 or " " in text or text.endswith(("다", "요", "습니다.", ".")):
        return {
            "mode": "sentence" if is_sentence else "term",
            "src_lang": "ko"
        }
    return {
        "mode": "term",
        "src_lang": "unknown"
    }


# ===============================================
# 용어 추출
# ===============================================

def extract_terms_from_sentence(sentence: str, min_len: int = 2, max_len: int = 8) -> list[str]:
    """
    문장에서 용어 후보 추출 (n-gram 방식)
    
    처리 방식:
    - 중국어: 연속된 한자 추출 후 n-gram 생성
    - 한국어: 공백 기준 분리 후 n-gram 생성
    
    Args:
        sentence: 입력 문장
        min_len: 최소 용어 길이 (기본: 2)
        max_len: 최대 용어 길이 (기본: 8)
    
    Returns:
        용어 후보 리스트 (중복 제거, 긴 것부터 정렬)
    
    예시:
        "发放10个超相尘" → ["超相尘", "超相", "相尘"]
        "월상석 가루 지급" → ["월상석 가루", "월상석", "가루", "지급"]
    """
    candidates = set()
    
    # 1. 중국어(한자) 연속 블록 추출 후 n-gram 생성
    cjk_blocks = re.findall(r"[\u4e00-\u9fff]+", sentence)
    for block in cjk_blocks:
        # n-gram 생성
        for n in range(min_len, min(len(block), max_len) + 1):
            for i in range(len(block) - n + 1):
                candidates.add(block[i:i+n])
    
    # 2. 한국어 단어 추출 (한글 연속 블록 기준)
    hangul_words = re.findall(r"[가-힣]+", sentence)
    for word in hangul_words:
        if min_len <= len(word) <= max_len:
            candidates.add(word)
    
    # 3. 한국어 복합어 추출 (공백으로 연결된 연속 단어 조합)
    # 예: "월상석 가루 10개 지급" → ["월상석 가루", "가루 지급", ...]
    if len(hangul_words) >= 2:
        for i in range(len(hangul_words)):
            for j in range(i + 1, min(i + 4, len(hangul_words) + 1)):  # 최대 3단어 조합
                compound = " ".join(hangul_words[i:j])
                if min_len <= len(compound) <= max_len + 4:  # 공백 포함하여 약간 여유
                    candidates.add(compound)
    
    # 긴 용어부터 정렬 (긴 용어가 더 구체적이므로 우선)
    return sorted(candidates, key=len, reverse=True)

