import requests, re

from doc_translate.google_authentification import get_access_token
from utils import get_translator_logger


logger = get_translator_logger("doc_translate.google_spreadsheets")


def get_spreadsheets_read(SHEET_URL: str, RANGE: str):
    """
    Google Sheets read API 호출
    Args:
        SHEET_URL (str): Google Sheets URL
        RANGE (str): 읽을 범위 (예: "A1:D10")
    Returns:
        dict: Google Sheets API 응답 (JSON)
    """
    
    # 1. Sheet URL 정규식 매칭
    sheet_url_normalized = re.search(r"/d/([a-zA-Z0-9-_]+)", SHEET_URL)
    if not sheet_url_normalized:
        logger.error(f"Invalid sheet URL: {SHEET_URL}")
        return None
    
    # 2. Sheet ID 추출
    sheet_id = sheet_url_normalized.group(1)
    
    # 3. Google Sheets API 호출
    access_token = get_access_token()
    headers = {
        "Authorization": access_token,
        "Content-Type": "application/json",
    }
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values:batchGet"
    payload = {
        "ranges": RANGE,
        "majorDimension": "ROWS",
    }
    response = requests.get(url, headers=headers, params=payload)
    logger.info(f"===== Google Sheets API RAW Response =====\n{response.text}")
    
    try:
        return response.json()
    except Exception as e:
        logger.error(f"JSON Parsing Error: {e}")
        return None
    

def put_spreadsheets_write(SHEET_URL: str, RANGE: str, VALUES: list):
    
    # 1. Sheet URL 정규식 매칭
    sheet_url_normalized = re.search(r"/d/([a-zA-Z0-9-_]+)", SHEET_URL)
    if not sheet_url_normalized:
        logger.error(f"Invalid sheet URL: {SHEET_URL}")
        return None
    
    # 2. Sheet ID 추출
    sheet_id = sheet_url_normalized.group(1)
    
    # 3. Google Sheets API 호출
    access_token = get_access_token()
    headers = {
        "Authorization": access_token,
        "Content-Type": "application/json",
    }
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/{RANGE}?valueInputOption=RAW"
    response = requests.put(url, headers=headers, json={"values": VALUES})
    logger.info(f"===== Google Sheets API RAW Response =====\n{response.text}")
    
    try:
        return response.json()
    except Exception as e:
        logger.error(f"JSON Parsing Error: {e}")
        return None
    
    