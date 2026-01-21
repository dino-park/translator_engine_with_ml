"""
디버깅용 번역 CLI
- 대화형 모드로 번역 테스트
- 인덱싱 모드 지원
"""

import argparse
import time

from utils import setup_logging, get_translator_logger
from engine import init_engine, translate_execute
from core.config import load_env, setup_settings
from indexing import run_indexing, run_indexing_for_text
from translation_logger_db import TranslationLoggerDB


logger = get_translator_logger("debug_cli")


def _init_query_context():
    """
    번역 쿼리 모드 초기화 공통 함수: 엔진 + DB로거
    Returns:
        translation_logger_db: 번역 로그 DB
    """
    # 엔진 초기화
    init_engine()
    
    # DB 로거 초기화 (source="debug"로 구분)
    translation_logger_db = TranslationLoggerDB()
    
    return translation_logger_db


# ===============================================
# 디버깅용 번역 쿼리 모드
# ===============================================

def run_query_mode(initial_query: str | None = None):
    """
    대화형 번역 쿼리 모드
    - initial_query가 있으면 해당 쿼리만 실행 후 종료
    - 없으면 대화형 모드로 진입
    """
    logger.info("[query] start. initial_query=%r", initial_query)
    translation_logger_db = _init_query_context()
    
    
    def handle_query(raw_query: str):
        """단일 쿼리 처리 및 결과 출력"""
        result = translate_execute(raw_query)
        
        # ML 학습용 데이터 로깅
        translation_logger_db.log_translation(result, source="debug")
        
        print(f"\n{'='*50}")
        print(f"입력: {result.get('raw_query', raw_query)}")
        print(f"번역: {result.get('translation', '(없음)')}")
        print(f"모드: {result.get('mode', '-')} | 언어: {result.get('src_lang', '-')}")
        print(f"방법: {result.get('reason', '-')}")
        
        # 후보 출력 (term 모드)
        if result.get('candidates'):
            print(f"후보: {result['candidates'][:3]}")
        
        # glossary 출력 (sentence 모드)
        if result.get('glossary'):
            print(f"용어사전: {result['glossary']}")
        
        # TM 매칭 정보 출력
        if result.get('tm_match'):
            tm = result['tm_match']
            print(f"TM: {tm.get('match_type', '-')} (유사도: {tm.get('similarity', 0)}%)")
        
        print(f"{'='*50}\n")
        return result
    
    # 단일 쿼리 모드
    if initial_query:
        handle_query(initial_query)
        return
    
    # 대화형 모드
    print("\n" + "="*50)
    print("  번역 시스템 (종료: q/quit/exit)")
    print("="*50 + "\n")
    
    while True:
        try:
            q = input("번역할 텍스트: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        
        if q.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            break
        if not q:
            continue
        
        handle_query(q)
    
    print("[query] 번역 쿼리 모드가 종료되었습니다.")


def run_sheet_query_mode(
    sheet_url: str, 
    range: str,
    output_column: str | None = None,
    dry_run: bool = False,
    ):
    
    from doc_translate.google_spreadsheets import get_spreadsheets_read, put_spreadsheets_write
    
    logger.info("[sheet query] start. sheet_url=%r, range=%r", sheet_url, range)
    # 1. 엔진 + DB로거 초기화
    translation_logger_db = _init_query_context()
    
    # 2. Google Sheets 읽기
    response = get_spreadsheets_read(sheet_url, range)
    if not response or "valueRanges" not in response:
        logger.error("[sheet query] failed to read Google Sheets")
        return
    
    rows = response["valueRanges"][0].get("values", [])
    if not rows:
        logger.warning("[sheet query] No data found in the sheet range.")
        return
    
    logger.info("[sheet query] Successfully. %d rows loaded", len(rows))
    
    # 3. 배치 번역 처리
    results = []
    success_count = 0
    fail_count = 0
    
    for i, row in enumerate(rows):
        if not row or not row[0].strip():
            results.append("")          # 빈 행 스킵
            continue
        
        raw_query = row[0].strip()
        
        try:
            result = translate_execute(raw_query)
            translation = result.get("translation", "")
            
            # ML 학습용 데이터 로깅
            translation_logger_db.log_translation(result, source="debug_sheet")
            
            results.append(translation)
            success_count += 1
            time.sleep(2.0)      # 2초 딜레이 (API 호출 제한 방지)
            
            if (i + 1) % 10 ==0:
                print(f"    처리 중...{i + 1}/{len(rows)}")
        
        except Exception as e:
            logger.error(f"[sheet query] row {i} translation failed: {e}")
            results.append(f"[Error] {e}")
            fail_count += 1
            time.sleep(2.0)      # 2초 딜레이 (API 호출 제한 방지)
    
    # 4. 결과 Report 출력
    print(f"\n{'='*50}")
    print(f"✅ 성공: {success_count}건")
    print(f"❌ 실패: {fail_count}건")
    print(f"{'='*50}")
    
    if output_column and not dry_run:
        # range에서 sheet 이름과 시작 행 추출
        import re
        match_name = re.match(r"(.+?)!([A-Z]+)(\d+)", range)
        if match_name:
            sheet_name, _, start_row = match_name.groups()
            end_row = int(start_row) + len(results) - 1
            output_range = f"{sheet_name}!{output_column}{start_row}:{output_column}{end_row}"
            updated_values = [[t] for t in results]
            put_spreadsheets_write(sheet_url, output_range, updated_values)
            
            print(f"📝 결과가 {output_column}열에 저장되었습니다.")
        else:
            logger.error(f"[sheet query] range parsing failed: {range}")
            print("⚠️ 출력 범위 파싱 실패. 결과 저장을 건너뜁니다.")
    elif dry_run:
        print("[dry_run] Sheet write를 skip합니다.")
        
        print("\n[미리보기 - 처음 5개]")
        for i, (row, trans) in enumerate(zip(rows[:5], results[:5])):
            src = row[0] if row else ""
            print(f" {src} -> {trans}")
    
    logger.info(f"[sheet query] completed. success={success_count}, fail={fail_count}")
            

def main():
    """CLI 메인 진입점"""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Mobile game CN↔KO translation system (Debug CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 CSV 인덱싱 (기존 데이터 유지 + 추가)
  python translator_for_debug.py --mode index_base
  
  # 기본 CSV 인덱싱 (기존 데이터 삭제 후 새로 인덱싱)
  python translator_for_debug.py --mode index_base --clear
  
  # 텍스트 인덱싱
  python translator_for_debug.py --mode index_text --cn "超现象管理局" --ko "초현상 관리국"
  
  # Google Sheets 번역
  python translator_for_debug.py --mode sheet --sheet_url "https://docs.google.com/spreadsheets/d/1234567890/edit#gid=0" --range "A1:B10" --output_column "C"
  
  # 단일 쿼리 번역
  python translator_for_debug.py --mode query --query "超现象管理局"
  
  # 대화형 번역 모드
  python translator_for_debug.py --mode query
        """
    )
    parser.add_argument(
        "--mode",
        choices=["index_base", "index_text", "query", "sheet"],
        required=True,
        help="index_base: 기본 CSV / index_text: 텍스트 인덱싱 / query: 번역 / sheet: Google Sheets"
    )
    parser.add_argument(
        "--sheet_url", 
        help="Google Sheets URL (mode=sheet 시 필요)"
    )
    parser.add_argument(
        "--range",
        help="읽을 범위 (mode=sheet 시 필요)"
    )
    parser.add_argument(
        "--output_column",
        help="출력 열 (mode=sheet 시 필요)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Google Sheets 쓰기 건너뜀 (mode=sheet 시)"
    )
    parser.add_argument(
        "--cn",
        help="인덱싱할 중국어 텍스트 (mode=index_text 시)"
    )
    parser.add_argument(
        "--ko",
        help="인덱싱할 한국어 텍스트 (mode=index_text 시)"
    )
    parser.add_argument(
        "--query",
        help="번역할 텍스트 (mode=query 시, 없으면 대화형 모드)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="기존 인덱스 삭제 후 새로 인덱싱 (mode=index_base 시)"
    )
    
    args = parser.parse_args()
    
    # 인덱싱 모드는 환경 설정 필요
    if args.mode in ("index_base", "index_text"):
        env = load_env()
        setup_settings(env)
    
    # 모드별 실행
    if args.mode == "index_base":
        run_indexing(clear=args.clear)
        print(f"[index_base] 완료 (clear={args.clear})")
        
    elif args.mode == "index_text":
        if not args.cn and not args.ko:
            raise ValueError("--cn 또는 --ko 옵션이 필요합니다 (mode=index_text)")
        run_indexing_for_text(cn=args.cn or "", ko=args.ko or "")
        print(f"[index_text] 완료: cn={args.cn}, ko={args.ko}")
        
    elif args.mode == "sheet":
        if not args.sheet_url or not args.range:
            raise ValueError("--sheet_url 과 --range 옵션이 필요합니다 (mode=sheet)")
        run_sheet_query_mode(
            sheet_url=args.sheet_url, 
            range=args.range,
            output_column=args.output_column,
            dry_run=args.dry_run,
            )
        
    elif args.mode == "query":
        run_query_mode(initial_query=args.query)


if __name__ == "__main__":
    main()
