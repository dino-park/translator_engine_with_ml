"""
기본 유틸리티 함수들
- 텍스트 언어 감지
- 공통 헬퍼 함수
- 로그 설정
"""

import re
import logging
import sys, threading

from datetime import datetime, date, timedelta
from logging.handlers import BaseRotatingHandler
from pathlib import Path


# ===== 명시적 로거 이름 상수 =====
# 새 모듈 추가 시 필터 수정 불필요 - 이 로거만 사용하면 됨

WEBHOOK_LOGGER_NAME = "app.webhook"       # webhook.log에 기록
TRANSLATOR_LOGGER_NAME = "app.translator"  # embedding_translator.log에 기록


class DailyRotatingFileHandler(BaseRotatingHandler):
    """
    날짜별 로그 파일 자동 생성 핸들러 클래스
    
    특징:
    - 파일명 형식: {base_name}_YYYY-MM-DD.log
    - 백그라운드 타이머로 자정에 자동 로테이션
    - 서버 재시작 시에도 올바른 날짜 파일에 기록
    """
    def __init__(self, log_dir:Path, base_name: str, encoding: str = 'utf-8'):
        self.log_dir = Path(log_dir)
        self.base_name = base_name
        self.encoding = encoding
        self._timer: threading.Timer = None
        
        self.log_dir.mkdir(exist_ok=True)
        
        # 현재 날짜 파일로 초기화
        filename = self._get_filename_for_date(date.today())
        super().__init__(filename, mode='a', encoding=encoding)
        
        # 자정 타이머 시작
        self._schedule_rollover()
    
    
    def _get_filename_for_date(self, d: date) -> str:
        return str(self.log_dir / f"{self.base_name}_{d.strftime('%Y-%m-%d')}.log")
    
    
    def _seconds_until_midnight(self) -> float:
        """다음 자정까지 남은 초 계산"""
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return (tomorrow - now).total_seconds()
    
    
    def _schedule_rollover(self):
        """다음 자정에 rollover 예약"""
        seconds = self._seconds_until_midnight()
        
        self._timer = threading.Timer(seconds, self._timer_rollover)
        self._timer.daemon = True       # 메인 프로세스 종료 시 함께 종료
        self._timer.start()
        
        
    def _timer_rollover(self):
        """타이머에 의해 자정에 호출됨"""
        with self.lock:                 # BaseRotatingHandler의 내장 Lock(스레드 안전)
            # 1. 현재 stream 닫기
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # 2. 새 파일명으로 Update
            self.baseFilename = self._get_filename_for_date(date.today())
            
            # 3. 새 파일 Stream 열기
            self.stream = self._open()
        
        # 4. 다음 자정 rollover 예약
        self._schedule_rollover()
    
    
    def shouldRollover(self, record) -> bool:
        return False                    # 항상 False 반환 -> 타이머가 처리하므로 체크 필요 없음
    
    
    def doRollover(self):
        pass                            # 미사용 - 타이머가 직접 처리
    
    
    def close(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None
        super().close()
        

def get_webhook_logger(name: str = None) -> logging.Logger:
    """
    웹훅/서버 관련 로거 반환 (webhook.log에 기록됨)
    
    Args:
        name: 선택적 서브 로거 이름 (예: "server" → "app.webhook.server")
    
    Usage:
        from utils import get_webhook_logger
        logger = get_webhook_logger()  # 또는 get_webhook_logger("server")
    """
    if name:
        return logging.getLogger(f"{WEBHOOK_LOGGER_NAME}.{name}")
    return logging.getLogger(WEBHOOK_LOGGER_NAME)


def get_translator_logger(name: str = None) -> logging.Logger:
    """
    번역 엔진 관련 로거 반환 (embedding_translator.log에 기록됨)
    
    Args:
        name: 선택적 서브 로거 이름 (예: "core" → "app.translator.core")
    
    Usage:
        from utils import get_translator_logger
        logger = get_translator_logger()  # 또는 get_translator_logger("engine")
    """
    if name:
        return logging.getLogger(f"{TRANSLATOR_LOGGER_NAME}.{name}")
    return logging.getLogger(TRANSLATOR_LOGGER_NAME)


# ===== 로그 분리용 필터 클래스 =====

class TranslatorLogFilter(logging.Filter):
    """
    번역 엔진 관련 로그만 통과시키는 필터
    - app.translator 로거와 그 하위 로거만 통과
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(TRANSLATOR_LOGGER_NAME)


class WebhookLogFilter(logging.Filter):
    """
    웹훅/서버 관련 로그만 통과시키는 필터
    - app.webhook 로거와 그 하위 로거, uvicorn, fastapi 등
    """
    # 외부 라이브러리 로거 (uvicorn, fastapi 등)
    EXTERNAL_LOGGERS = ("uvicorn", "fastapi", "httpx", "httpcore")
    
    def filter(self, record: logging.LogRecord) -> bool:
        # app.webhook 로거
        if record.name.startswith(WEBHOOK_LOGGER_NAME):
            return True
        
        # 외부 라이브러리 로거
        for prefix in self.EXTERNAL_LOGGERS:
            if record.name == prefix or record.name.startswith(f"{prefix}."):
                return True
        
        # root 로거
        if record.name == "root" or record.name == "":
            return True
        
        return False


# ===== 로깅 설정 함수 =====

def setup_logging():
    """
    번역 엔진용 로깅 설정 (embedding_translator.log)
    - Only CLI 모드(translator_for_debug.py)에서 호출됨
    - 기존 설정에 embedding_translator.log FileHandler 추가
    """
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "embedding_translator.log"
    
    root_logger = logging.getLogger()
    
    # 이미 동일한 파일 핸들러가 있는지 확인(DailyRotating 타입 체크)
    for handler in root_logger.handlers:
        if isinstance(handler, DailyRotatingFileHandler):
            if handler.base_name == "embedding_translator":
                return  # 이미 추가됨
    
    # embedding_translator.log 용 DailyRotatingFileHandler 추가
    file_handler = DailyRotatingFileHandler(log_dir=log_dir, base_name="embedding_translator")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"))
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # basicConfig가 아직 안 되어 있으면 기본 설정
    if not root_logger.handlers or len(root_logger.handlers) == 1:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"))
        root_logger.addHandler(stream_handler)
        root_logger.setLevel(logging.DEBUG)


def setup_server_logging():
    """
    서버 모드용 로깅 설정 (로그 분리 + 날짜별 로테이션)
    - webhook.log: 웹훅/서버 관련 로그만 기록
    - embedding_translator.log: 번역 엔진 관련 로그만 기록
    - 콘솔: 모든 로그 출력
    """
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)
    
    log_format = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 모두 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 1. 콘솔 출력 (모든 로그)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    
    # 2. webhook.log (웹훅/서버 관련만, 날짜별 로테이션 파일 분리 적용)
    webhook_handler = DailyRotatingFileHandler(log_dir=log_dir, base_name="webhook")
    webhook_handler.setFormatter(log_format)
    webhook_handler.setLevel(logging.INFO)
    webhook_handler.addFilter(WebhookLogFilter())
    root_logger.addHandler(webhook_handler)
    
    # 3. embedding_translator.log (번역 엔진만, 날짜별 로테이션 파일 분리 적용)
    translator_handler = DailyRotatingFileHandler(log_dir=log_dir, base_name="embedding_translator")
    translator_handler.setFormatter(log_format)
    translator_handler.setLevel(logging.DEBUG)
    translator_handler.addFilter(TranslatorLogFilter())
    root_logger.addHandler(translator_handler)
    
    logging.info("Server logging configured with daily rotation.")


def has_cjk(text: str) -> bool:
    """
    텍스트에 중국어(한자)가 포함되어 있는지 확인
    
    Args:
        text: 검사할 텍스트
    
    Returns:
        True if 중국어 포함, False otherwise
    
    예시:
        has_cjk("超现象管理局") → True
        has_cjk("초현상 관리국") → False
        has_cjk("超现象 관리국") → True
    """
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def has_hangul(text: str) -> bool:
    """
    텍스트에 한글이 포함되어 있는지 확인
    
    Args:
        text: 검사할 텍스트
    
    Returns:
        True if 한글 포함, False otherwise
    
    예시:
        has_hangul("초현상 관리국") → True
        has_hangul("超现象管理局") → False
        has_hangul("超现象 관리국") → True
    """
    return bool(re.search(r"[\uac00-\uD7A3]", text))


def is_date_or_version_pattern(text: str) -> bool:
    """
    텍스트가 날짜 또는 버전 패턴인지 확인
    
    번역할 필요가 없는 패턴들:
    - 날짜: 2025.12.10, 2025-12-10, 2025/12/10
    - 버전: 1.1.189, v2.0.1, Ver 1.2.3
    - 시간: 12:30, 14:00:00
    - 숫자만: 12345 (5자리 이상)
    
    Args:
        text: 검사할 텍스트
    
    Returns:
        True if 번역 스킵 대상 패턴, False otherwise
    
    예시:
        is_date_or_version_pattern("2025.12.10") → True
        is_date_or_version_pattern("1.1.189") → True
        is_date_or_version_pattern("v2.0.1") → True
        is_date_or_version_pattern("12:30") → True
        is_date_or_version_pattern("12345") → True (5자리 이상 숫자)
        is_date_or_version_pattern("게임 시작") → False
        is_date_or_version_pattern("1.5") → False (너무 짧음)
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # 패턴 1: 날짜 (YYYY.MM.DD, YYYY-MM-DD, YYYY/MM/DD 등)
    # 예: 2025.12.10, 2025-01-28, 2025/12/31
    date_pattern = r'^\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}$'
    if re.match(date_pattern, text):
        return True
    
    # 패턴 2: 버전 (X.Y.Z 형식, v 접두사 포함)
    # 예: 1.1.189, v2.0.1, Ver 1.2.3, version 1.0
    # Ver와 숫자 사이 공백 허용
    version_pattern = r'^(v|ver|version)\s*\d+\.\d+(\.\d+)*$'
    if re.match(version_pattern, text, re.IGNORECASE):
        return True
    
    # 패턴 2-1: 순수 버전 번호 (접두사 없음, 3개 이상 숫자)
    # 예: 1.1.189 (O), 1.5 (X - 너무 짧음)
    pure_version_pattern = r'^\d+\.\d+\.\d+(\.\d+)*$'
    if re.match(pure_version_pattern, text):
        return True
    
    # 패턴 3: 시간 (HH:MM 또는 HH:MM:SS)
    # 예: 12:30, 14:00:00
    time_pattern = r'^\d{1,2}:\d{2}(:\d{2})?$'
    if re.match(time_pattern, text):
        return True
    
    # 패턴 4: 순수 숫자만 (5자리 이상)
    # 예: 12345, 100000
    # 너무 짧은 숫자는 제외 (1, 2, 12 등은 번역 가능한 용어일 수 있음)
    number_pattern = r'^\d{5,}$'
    if re.match(number_pattern, text):
        return True
    
    # 패턴 5: 년-월 형식 (YYYY.MM, YYYY-MM)
    # 예: 2025.12, 2025-01
    year_month_pattern = r'^\d{4}[.\-]\d{1,2}$'
    if re.match(year_month_pattern, text):
        return True
    
    return False