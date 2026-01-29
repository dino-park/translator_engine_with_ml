
import re, time

from doc_translate.google_spreadsheets import get_spreadsheets_read, put_spreadsheets_write
from utils import get_translator_logger, is_date_or_version_pattern

logger = get_translator_logger("doc_translate.sheet_translator")


def translate_sheet(
    sheet_url: str,
    range: str,
    output_column: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Google Sheets 배치 번역 함수
    Args:
        sheet_url (str): Google Sheets URL
        range (str): 읽을 범위 (예: "A1:D10")
        output_column (str | None): 출력 열 (예: "C")
        progress_callback: 진행 상태 콜백 함수(async 함수, Optional)

    Returns:
        dict: 번역 결과 (success_count, fail_count, total_count, results, error)
    """
    
    from engine import translate_execute
    
    logger.info(f"[Sheet Translator] start: url={sheet_url}, range={range}")
    
    # 1. Google Sheets 읽기
    response = get_spreadsheets_read(sheet_url, range)
    if not response or "valueRanges" not in response:
        logger.error("[sheet_translator] Failed to read sheet")
        return {
            "success_count": 0,
            "fail_count": 0,
            "total_count": 0,
            "results": [],
            "error": "시트 읽기 실패. URL 또는 권한을 확인하세요."
        }
    
    rows = response["valueRanges"][0].get("values", [])
    if not rows:
        logger.warning("[sheet_translator] No data found in the sheet range.")
        return {
            "success_count": 0,
            "fail_count": 0,
            "total_count": 0,
            "results": [],
            "error": "해당 범위에 데이터가 없습니다."
        }
    
    total_count = len(rows)
    logger.info(f"[sheet_translator] Successfully loaded {total_count} rows from the sheet range.")
    
    # 2. 배치 번역 처리
    results = []
    translation_results = []            # DB 배치 저장용
    success_count = 0
    fail_count = 0
    
    for i, row in enumerate(rows):
        if not row or not row[0].strip():
            results.append("")          # 빈 행 스킵
            continue
        
        raw_query = row[0].strip()
        
        # 날짜/버전 패턴 체크 (번역 스킵)
        if is_date_or_version_pattern(raw_query):
            logger.info(f"[sheet_translator] Row {i}: Skipping date/version pattern - %r", raw_query)
            results.append(raw_query)  # 원문 그대로
            success_count += 1
            continue
        
        try:
            result = translate_execute(raw_query)
            translation = result.get("translation", "")
            results.append(translation)
            success_count += 1
            
            # DB 배치 저장을 위해 리스트에 추가
            translation_results.append(result)
        
        except Exception as e:
            logger.error(f"[sheet_translator] Row {i} translation failed: {e}")
            results.append("")
            fail_count += 1
        
        time.sleep(0.5)      # 0.5초 딜레이 (API 호출 제한 방지)
        
        if progress_callback and (i + 1) % 10 == 0:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(progress_callback):
                    asyncio.create_task(progress_callback(i + 1, total_count))
            except:
                pass
            
    if translation_results:
        from translation_logger_db import translation_logger_db
        logger.info("[sheet_translator] Saving %d results to DB (batch mode)", len(translation_results))
        translation_logger_db.log_translation_batch(translation_results, source="api")
        logger.info("[sheet_translator] DB batch save completed")
            
    logger.info(f"[sheet_translator] Translation done: success={success_count}, fail={fail_count}")
    
    # 3. 결과 쓰기 (Optional)
    if output_column and results:
        # range에서 sheet 이름과 시작 행 추출
        match_name = re.match(r"(.+?)!([A-Z]+)(\d+)", range)
        if match_name:
            sheet_name, _, start_row = match_name.groups()
            end_row = int(start_row) + len(results) - 1
            output_range = f"{sheet_name}!{output_column}{start_row}:{output_column}{end_row}"
            updated_values = [[t] for t in results]
            
            write_response = put_spreadsheets_write(sheet_url, output_range, updated_values)
            if write_response:
                logger.info(f"[sheet_translator] Successfully written to column {output_column}")
            else:
                logger.error(f"[sheet_translator] Failed to write to column {output_column}")
        else:
            logger.error(f"[sheet_translator] Range parsing failed: {range}")
    
    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "total_count": total_count,
        "results": results,
        "error": None,
    }
