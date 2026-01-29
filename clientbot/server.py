from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import os, json, hmac, hashlib, re

from pydantic.type_adapter import R

from clientbot.seatalk_service.seatalk_client import seatalk_client
from engine import init_engine, translate_execute
from translation_logger_db import translation_logger_db
from doc_translate.sheet_translator import translate_sheet
from utils import setup_server_logging, get_webhook_logger, is_date_or_version_pattern


load_dotenv()
SEATALK_SIGNING_SECRET = os.getenv("SEATALK_SIGNING_SECRET")

# 로그 분리 설정: webhook.log (서버) + embedding_translator.log (번역 엔진)
setup_server_logging()

# 서버 전용 로거 (webhook.log에 기록됨)
logger = get_webhook_logger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 번역 엔진 초기화"""
    logger.info("Initializing translation engine...")
    init_engine()
    logger.info("Translation engine initialized successfully")
    yield
    # 종료 시 정리 작업 (필요 시)
    logger.info("Shutting down...")


app = FastAPI(title="My Callback Server", lifespan=lifespan)


# ----- 서명 계산 함수 -----
def calc_signature(raw_body: bytes, secret: str) -> str:
    """
    SealTalk 스펙: (raw_body + secret) byte를 SHA256으로 hash -> hex소문자

    Args:
        raw_body (bytes): 원본 요청 바디
        secret (str): 서명 비밀키

    Returns:
        str: 생성된 서명 (hex 형식)
    """
    if not secret:
        return ""
    payload = raw_body + secret.encode("utf-8")    
    return hashlib.sha256(payload).hexdigest()


def consteq(a: str, b: str) -> bool:
    """ 안전 비교(대소문자 무시하지 않고, 스펙대로 hex 소문자 비교) """
    return hmac.compare_digest((a or "").strip(), (b or "").strip())

# ----- Seatalk event에서 사용자 텍스트 추출 -----
def get_user_text(event: dict, remove_mentions: bool = False) -> str:
    """
    Seatalk 콜백 event에서 text를 문자열로 추출
    - 1:1 채팅: event["message"]["text"]["content"]
    - 그룹챗 멘션: event["message"]["text"]["plain_text"]

    Args:
        event (dict): 이벤트 dict
        remove_mentions (bool): True이면 멘션된 봇/사용자 이름 제거

    Returns:
        str: 사용자 텍스트
    """
    msg = event.get("message") or {}
    text_field = msg.get("text") or {}
    
    if not isinstance(text_field, dict):
        return ""       # text 필드가 dict가 아닌 경우 빈 문자열 반환
    
    # 1:1 채팅
    if isinstance(text_field.get("content"), str):
        return text_field["content"]
    
    # 그룹챗 멘션
    if isinstance(text_field.get("plain_text"), str):
        text = text_field["plain_text"]
        
        # 멘션 제거 옵션이 켜져 있으면 @username 제거
        if remove_mentions:
            mentioned_list = text_field.get("mentioned_list") or []
            for mention in mentioned_list:
                username = mention.get("username", "")
                if username:
                    # "@username " 또는 "@username" 패턴 제거
                    text = re.sub(rf"@{re.escape(username)}\s*", "", text)
        
        return text.strip()
    
    return ""


#----- Google Sheets URL 정규식 -----
SHEETS_URL_RE = re.compile(
    r"(https?://docs\.google\.com/spreadsheets/(?:u/\d+/)?d/([a-zA-Z0-9-_]+)(?:/[^\s]*)?)",
    re.IGNORECASE
)
# 그룹: (1) URL, (2) Range, (3) Output Column(선택, A-Z 중 한 글자)
SHEETS_WITH_RANGE_RE = re.compile(
    r"(https?://docs\.google\.com/spreadsheets/[^\s]+)\s+([A-Za-z0-9_가-힣\u4e00-\u9fff]+![A-Z]+\d*:[A-Z]+\d*)(?:\s+(?:->)?\s*([A-Z]))?",
    re.IGNORECASE
)


def detect_google_sheet_url(text: str):
    """
    텍스트에서 Google Sheets URL을 감지하여 반환
    - URL, Range, Output Column
    Args:
        text (str): 입력 텍스트

    """
    if not text:
        return None, None, None
    
    # 1. URL + Range(+ Output Column) 패턴 매칭 먼저 시도
    match_with_range = SHEETS_WITH_RANGE_RE.search(text)
    if match_with_range:
        url = match_with_range.group(1)
        range = match_with_range.group(2)
        output_column = match_with_range.group(3)
        return url, range, output_column
    
    # 2. URL 패턴 매칭
    match_url_only = SHEETS_URL_RE.search(text)
    if match_url_only:
        return match_url_only.group(1), None, None
    return None, None, None


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}


async def process_group_message(group_id: str, user_text: str, message_id: str, thread_id: str):
    """그룹 메시지 처리 (백그라운드 태스크)"""
    sheet_url, range, output_column = detect_google_sheet_url(user_text)
    
    # case 1: URL + Range + Output Column 모두 있을 경우 -> 번역 실행
    if sheet_url and range and output_column:
        logger.info(f"[Group] Detected Google Sheets URL + Range + Output Column: {sheet_url}, {range}, {output_column}")
        
        await seatalk_client.send_group_text_message(
            group_id,
            f"🔄 번역을 시작합니다...\n📊 범위: {range}",
            message_id, thread_id
        )
        result = translate_sheet(
            sheet_url=sheet_url,
            range=range,
            output_column=output_column
        )
        
        if result["error"]:
            reply_text = f"❌ 번역 중 오류가 발생했습니다: {result['error']}"
            
        else:
            reply_text = (
                f"✅ 번역 완료!\n"
                f"📊 전체: {result['total_count']}건\n"
                f"✅ 성공: {result['success_count']}건\n"
                f"❌ 실패: {result['fail_count']}건"
            )
        await seatalk_client.send_group_text_message(group_id, reply_text, message_id, thread_id)
    
    # case 2: URL + range는 있으나, Output Column 없을 경우 -> 안내 메시지 전송
    elif sheet_url and range:
        logger.info(f"[Group] Detected Google Sheets URL + range (no output column): {sheet_url}, {range}")
        reply_text = (
            f"⭐Google Sheets 링크와 범위는 감지했어요!⭐\n\n"
            f"📌 출력 열을 추가로 입력해주세요:\n"
            f"e.g. {sheet_url} {range} C🔻"
        )
        await seatalk_client.send_group_text_message(group_id, reply_text, message_id, thread_id)

            
    # case 3: URL만 있을 경우 -> 안내 메시지 전송
    elif sheet_url:
        logger.info(f"[Group] Detected Google Sheets URL (no range and output column): {sheet_url}")
        reply_text = (
            f"⭐Google Sheets 링크를 감지했어요!⭐\n\n"
            f"📌 범위와 출력 열을 함께 입력해주세요:\n"
            f"e.g. {sheet_url} Sheet1!A2:A100 C🔻"
        )
        await seatalk_client.send_group_text_message(group_id, reply_text, message_id, thread_id)
        
    # case 4: 일반 번역
    else:
        if user_text.strip():
            # 날짜/버전 패턴 체크 (번역 스킵)
            if is_date_or_version_pattern(user_text):
                logger.info(f"[Group] Skipping translation: date/version pattern detected - %r", user_text)
                reply_text = f"📝 번역 결과:\n{user_text}"
            else:
                logger.info(f"[Group] No Google Sheets URL detected, calling translation engine")
                try:
                    result = translate_execute(user_text)
                    translation = result.get("translation")
                    
                    # ----- 번역 로그 저장 (ML 학습용) -----
                    translation_logger_db.log_translation(result, source="api")
                    
                    if translation:
                        reply_text = f"📝 번역 결과:\n{translation}"
                    else:
                        reason = result.get("reason", "unknown")
                        reply_text = f"❌ 번역할 수 없습니다. (사유: {reason})"
                except Exception as e:
                    logger.error(f"[Group] Translation error: {e}")
                    reply_text = f"⚠️ 번역 중 오류가 발생했습니다: {str(e)}"
        else:
            reply_text = ("💡번역할 텍스트를 입력해주세요.\n\n"
                          "또는 Google Sheets URL + Range를 입력하시면 문서 번역을 시작합니다.\n"
                          "e.g. https://docs.google.com/.../d/xxx Sheet1!A2:A100")
        await seatalk_client.send_group_text_message(group_id, reply_text, message_id, thread_id)
    
    logger.info(f"[Group] Processed: group_id={group_id}, message_id={message_id}")


async def process_single_message(employee_code: str, user_text: str, message_id: str):
    """1:1 메시지 처리 (백그라운드 태스크)"""
    sheet_url, range, output_column = detect_google_sheet_url(user_text)
    
    # case 1: URL + Range + Output Column 모두 있을 경우 -> 번역 실행
    if sheet_url and range and output_column:
        logger.info(f"Detected Google Sheets URL + Range + Output Column: {sheet_url}, {range}, {output_column}")
        # Start 알림
        reply_text = f"⭐Google Sheets 링크와 범위를 감지했어요!⭐\n{sheet_url}\n🔄 번역을 시작합니다...\n📊 범위: {range}"
        await seatalk_client.send_text_message(employee_code, reply_text)
        
        # 번역 실행
        result = translate_sheet(
            sheet_url=sheet_url,
            range=range,
            output_column=output_column,
        )
                
        if result["error"]:
            reply_text = f"❌ 번역 중 오류가 발생했습니다: {result['error']}"
        else:
            reply_text = (
                f"✅ 번역 완료!\n"
                f"📊 전체: {result['total_count']}건\n"
                f"✅ 성공: {result['success_count']}건\n"
                f"❌ 실패: {result['fail_count']}건"
            )
        await seatalk_client.send_text_message(employee_code, reply_text)
    
    # case 2: URL + range는 있으나, Output Column 없을 경우 -> 안내 메시지 전송
    elif sheet_url and range:
        logger.info(f"Detected Google Sheets URL + range (no output column): {sheet_url}, {range}")
        reply_text = (
            f"⭐Google Sheets 링크와 범위는 감지했어요!⭐\n\n"
            f"📌 출력 열을 추가로 입력해주세요:\n"
            f"e.g. {sheet_url} {range} C🔻"
        )
        await seatalk_client.send_text_message(employee_code, reply_text)
        
    # case 3: URL만 있을 경우 -> 안내 메시지 전송
    elif sheet_url:
        logger.info(f"Detected Google Sheets URL (no range and output column): {sheet_url}")
        reply_text = (
            f"⭐Google Sheets 링크를 감지했어요!⭐\n\n"
            f"📌 범위와 출력 열을 함께 입력해주세요:\n"
            f"e.g. {sheet_url} Sheet1!A2:A100 C🔻"
        )
        await seatalk_client.send_text_message(employee_code, reply_text)
        
    # case 3: 일반 번역
    else:
        if user_text.strip():
            # 날짜/버전 패턴 체크 (번역 스킵)
            if is_date_or_version_pattern(user_text):
                logger.info(f"Skipping translation: date/version pattern detected - %r", user_text)
                reply_text = f"📝 번역 결과:\n{user_text}"
            else:
                logger.info(f"No Google Sheets URL detected, calling translation engine")
                try:
                    result = translate_execute(user_text)
                    translation = result.get("translation")
                    
                    # ----- 번역 로그 저장 (ML 학습용) -----
                    translation_logger_db.log_translation(result, source="api")
                    
                    if translation:
                        reply_text = f"📝 번역 결과:\n{translation}"
                    else:
                        reason = result.get("reason", "unknown")
                        reply_text = f"❌ 번역할 수 없습니다. (사유: {reason})"
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    reply_text = f"⚠️ 번역 중 오류가 발생했습니다: {str(e)}"
        else:
            reply_text = (
                "💡 번역할 텍스트를 입력해주세요.\n\n"
                "또는 Google Sheets URL + Range를 입력하시면 문서 번역을 시작합니다.\n"
                "e.g. https://docs.google.com/.../d/xxx Sheet1!A2:A100"
            )
        await seatalk_client.send_text_message(employee_code, reply_text)
    
    logger.info(f"[Single] Processed: employee_code={employee_code}, message_id={message_id}")


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks, signature_hdr: str | None = Header(default=None, alias="Signature")):
    """
    외부 서비스가 이 URL로 Post를 보냄.
    - request.json(): JSON 본문
    - x-signature: 서비스가 보내는 서명 헤더 예시

    Args:
        request (Request): _description_
        x_signature (str | None, optional): _description_. Defaults to Header(default=None).
    """
    # ----- 본문 파싱 -----
    raw = await request.body()
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        # 검증 자체는 raw로 하지만, 파싱 실패 시에도 400으로 처리
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    event_type = body.get("event_type")
    event = body.get("event") or {}
    
    # - callback URL 검증용 처리
    if event_type == "event_verification":
        challenge = event.get("seatalk_challenge")
        if not challenge:
            raise HTTPException(status_code=400, detail="Missing seatalk_challenge")
        
        if signature_hdr and SEATALK_SIGNING_SECRET:
            # 서명 검증
            expected = calc_signature(raw, SEATALK_SIGNING_SECRET)
            if not consteq(signature_hdr.lower(), expected.lower()):
                # 검증 실패를 로깅만 하고 에코는 수행함.
                pass
        # Spec: 받은 값 그대로 전달
        return JSONResponse({"seatalk_challenge": challenge}, status_code=200)
    
    if not (signature_hdr and SEATALK_SIGNING_SECRET):
        raise HTTPException(status_code=401, detail="Missing Signature or secret")
    
    expected = calc_signature(raw, SEATALK_SIGNING_SECRET)
    if not consteq(signature_hdr.lower(), expected.lower()):
        raise HTTPException(status_code=401, detail="Invalid Signature")
    
    print("\n========== [SeaTalk Webhook Received] ==========")
    logger.info(f"Headers: {dict(request.headers)}")
    logger.info(f"Body: {body}")
    print("=================================================\n")
    
    if event_type == "message_from_bot_subscriber":
        employee_code = event.get("employee_code")
        message_id = event.get("message", {}).get("message_id")
        user_text = get_user_text(event)
        
        # 백그라운드에서 처리하고 즉시 응답 반환 (SeaTalk 타임아웃 방지)
        background_tasks.add_task(process_single_message, employee_code, user_text, message_id)
        logger.info(f"[Single] Queued: employee_code={employee_code}, message_id={message_id}, text='{user_text}'")
    
    elif event_type == "new_mentioned_message_received_from_group_chat":
        group_id = event.get("group_id")
        message_id = event.get("message", {}).get("message_id")
        thread_id = event.get("message", {}).get("thread_id") or None
        user_text = get_user_text(event, remove_mentions=True)  # 멘션된 봇 이름 제거
        
        # 백그라운드에서 처리하고 즉시 응답 반환 (SeaTalk 타임아웃 방지)
        background_tasks.add_task(process_group_message, group_id, user_text, message_id, thread_id)
        logger.info(f"[Group] Queued: group_id={group_id}, message_id={message_id}, thread_id={thread_id}")
    
    return JSONResponse({"received": True}, status_code=200)
