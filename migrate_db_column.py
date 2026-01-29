# migrate_db_column.py
"""
데이터베이스 컬럼명 변경 스크립트
bm25_hybrid_rank → bm25_top_rank_in_hybrid
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "database" / "translation_logger.db"

def migrate_column_name():
    """
    SQLite는 컬럼명 변경을 직접 지원하지 않으므로
    1. 새 테이블 생성
    2. 데이터 복사
    3. 기존 테이블 삭제
    4. 테이블 이름 변경
    """
    print(f"DB 경로: {DB_PATH}")
    
    if not DB_PATH.exists():
        print("⚠️ DB 파일이 없습니다. 새로 생성될 때 올바른 컬럼명이 사용됩니다.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. 현재 테이블 정보 확인
    cursor.execute("PRAGMA table_info(translation_logger)")
    columns = cursor.fetchall()
    
    # bm25_hybrid_rank 컬럼이 있는지 확인
    has_old_column = any(col[1] == "bm25_hybrid_rank" for col in columns)
    has_new_column = any(col[1] == "bm25_top_rank_in_hybrid" for col in columns)
    
    if not has_old_column:
        if has_new_column:
            print("✅ 이미 bm25_top_rank_in_hybrid 컬럼을 사용 중입니다.")
        else:
            print("⚠️ bm25_hybrid_rank 컬럼이 없습니다. 마이그레이션이 필요없습니다.")
        conn.close()
        return
    
    print("🔧 컬럼명 변경 시작...")
    
    try:
        # 2. 백업 테이블 생성
        cursor.execute("""
            CREATE TABLE translation_logger_backup AS 
            SELECT * FROM translation_logger
        """)
        print("  ✓ 백업 생성")
        
        # 3. 기존 테이블 삭제
        cursor.execute("DROP TABLE translation_logger")
        print("  ✓ 기존 테이블 삭제")
        
        # 4. 새 테이블 생성 (올바른 컬럼명으로)
        cursor.execute("""
            CREATE TABLE translation_logger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- 입력 정보
                query TEXT,
                query_len INTEGER,
                src_lang TEXT,
                mode TEXT,
                
                -- 출력 정보
                translation TEXT,
                reason TEXT,
                
                -- 검색 점수 (ML 피처로 사용)
                top_score REAL,
                candidate_gap REAL,
                candidates_json TEXT,
                
                -- 매칭 유형 (ML 피처 + Weak Label 생성용)
                is_exact_match BOOLEAN,
                is_bm25_match BOOLEAN,
                is_llm_fallback BOOLEAN,
                
                -- BM25 분석 정보 (Vector vs BM25 비교)
                bm25_exact_rank INTEGER,
                bm25_top_rank_in_hybrid INTEGER,
                
                -- segment/metadata 기반 feature
                top_doc_type TEXT,
                is_segment_exact_match INTEGER,
                segment_parent_cn TEXT,
                
                -- glossary hints 관련 (1단계 대응)
                has_glossary_hints BOOLEAN,
                glossary_match_count INTEGER,
                
                -- 신뢰도 체크 관련 (1단계 대응)
                passed_bm25_check BOOLEAN,
                passed_gap_check BOOLEAN,
                
                -- 성능 관련
                response_time_ms INTEGER,
                
                -- 메타 정보
                source TEXT DEFAULT 'api'
            )
        """)
        print("  ✓ 새 테이블 생성")
        
        # 5. 데이터 복사 (컬럼명 매핑)
        cursor.execute("""
            INSERT INTO translation_logger 
            SELECT 
                id, created_at, query, query_len, src_lang, mode,
                translation, reason, top_score, candidate_gap, candidates_json,
                is_exact_match, is_bm25_match, is_llm_fallback,
                bm25_exact_rank, 
                bm25_hybrid_rank,  -- 기존 컬럼명
                top_doc_type, is_segment_exact_match, segment_parent_cn,
                has_glossary_hints, glossary_match_count,
                passed_bm25_check, passed_gap_check,
                response_time_ms, source
            FROM translation_logger_backup
        """)
        print(f"  ✓ 데이터 복사 완료 ({cursor.rowcount}개 행)")
        
        # 6. 백업 테이블 삭제
        cursor.execute("DROP TABLE translation_logger_backup")
        print("  ✓ 백업 삭제")
        
        conn.commit()
        print("✅ 마이그레이션 완료!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ 에러 발생: {e}")
        print("  롤백 중...")
        
        # 백업이 있다면 복원 시도
        try:
            cursor.execute("DROP TABLE IF EXISTS translation_logger")
            cursor.execute("ALTER TABLE translation_logger_backup RENAME TO translation_logger")
            conn.commit()
            print("  ✓ 백업에서 복원 완료")
        except:
            print("  ❌ 복원 실패 - 수동으로 백업 테이블 확인 필요")
    
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_column_name()
    