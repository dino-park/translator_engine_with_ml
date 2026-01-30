# migrate_db_column.py
"""
데이터베이스 마이그레이션 유틸리티
- 컬럼 추가
- 컬럼 이름 변경
- 범용적으로 사용 가능한 마이그레이션 함수 제공
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent / "database" / "translation_logger.db"
TABLE_NAME = "translation_logger"


def get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[tuple]:
    """테이블의 컬럼 정보 조회"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()


def column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    """특정 컬럼이 존재하는지 확인"""
    columns = get_table_columns(cursor, table_name)
    return any(col[1] == column_name for col in columns)


def add_column(
    table_name: str = TABLE_NAME,
    column_name: str = None,
    column_type: str = "TEXT",
    default_value: Any = None,
    nullable: bool = True,
    db_path: Path = DB_PATH
) -> bool:
    """
    테이블에 새로운 컬럼 추가
    
    Args:
        table_name: 테이블 이름
        column_name: 추가할 컬럼 이름
        column_type: 컬럼 타입 (TEXT, INTEGER, REAL, BOOLEAN 등)
        default_value: 기본값 (None이면 NULL)
        nullable: NULL 허용 여부
        db_path: 데이터베이스 파일 경로
    
    Returns:
        성공 여부
    """
    if not column_name:
        print("❌ 컬럼 이름을 지정해주세요.")
        return False
    
    print(f"DB 경로: {db_path}")
    
    if not db_path.exists():
        print("⚠️ DB 파일이 없습니다.")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 컬럼 존재 여부 확인
        if column_exists(cursor, table_name, column_name):
            print(f"✅ '{column_name}' 컬럼이 이미 존재합니다.")
            return True
        
        print(f"🔧 '{column_name}' 컬럼 추가 시작...")
        
        # ALTER TABLE로 컬럼 추가
        column_def = f"{column_name} {column_type}"
        
        if default_value is not None:
            if isinstance(default_value, str):
                column_def += f" DEFAULT '{default_value}'"
            else:
                column_def += f" DEFAULT {default_value}"
        
        if not nullable:
            column_def += " NOT NULL"
        
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")
        
        conn.commit()
        print(f"✅ '{column_name}' 컬럼 추가 완료!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ 에러 발생: {e}")
        return False
    
    finally:
        conn.close()


def rename_column(
    table_name: str = TABLE_NAME,
    old_column_name: str = None,
    new_column_name: str = None,
    db_path: Path = DB_PATH
) -> bool:
    """
    컬럼 이름 변경 (SQLite는 ALTER TABLE RENAME COLUMN을 지원하지만,
    복잡한 경우 테이블 재생성 방식 사용)
    
    Args:
        table_name: 테이블 이름
        old_column_name: 기존 컬럼 이름
        new_column_name: 새 컬럼 이름
        db_path: 데이터베이스 파일 경로
    
    Returns:
        성공 여부
    """
    if not old_column_name or not new_column_name:
        print("❌ 기존 컬럼명과 새 컬럼명을 지정해주세요.")
        return False
    
    print(f"DB 경로: {db_path}")
    
    if not db_path.exists():
        print("⚠️ DB 파일이 없습니다.")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 컬럼 존재 여부 확인
        has_old = column_exists(cursor, table_name, old_column_name)
        has_new = column_exists(cursor, table_name, new_column_name)
        
        if not has_old:
            if has_new:
                print(f"✅ 이미 '{new_column_name}' 컬럼을 사용 중입니다.")
            else:
                print(f"⚠️ '{old_column_name}' 컬럼이 없습니다.")
            return True
        
        print(f"🔧 컬럼명 변경 시작: {old_column_name} → {new_column_name}")
        
        # SQLite 3.25.0 이상에서 지원하는 RENAME COLUMN 사용
        cursor.execute(f"ALTER TABLE {table_name} RENAME COLUMN {old_column_name} TO {new_column_name}")
        
        conn.commit()
        print("✅ 컬럼명 변경 완료!")
        return True
        
    except sqlite3.OperationalError as e:
        # RENAME COLUMN을 지원하지 않는 경우 테이블 재생성 방식 사용
        print(f"  ⚠️ RENAME COLUMN 미지원: {e}")
        print("  🔄 테이블 재생성 방식으로 시도...")
        conn.rollback()
        
        return _rename_column_with_table_recreation(
            conn, cursor, table_name, old_column_name, new_column_name
        )
    
    except Exception as e:
        conn.rollback()
        print(f"❌ 에러 발생: {e}")
        return False
    
    finally:
        conn.close()


def _rename_column_with_table_recreation(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    table_name: str,
    old_column_name: str,
    new_column_name: str
) -> bool:
    """
    테이블 재생성을 통한 컬럼 이름 변경
    (RENAME COLUMN을 지원하지 않는 구버전 SQLite용)
    """
    try:
        # 현재 테이블 스키마 가져오기
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_sql = cursor.fetchone()[0]
        
        # 백업 생성
        backup_table = f"{table_name}_backup"
        cursor.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}")
        print(f"  ✓ 백업 생성")
        
        # 기존 테이블 삭제
        cursor.execute(f"DROP TABLE {table_name}")
        print(f"  ✓ 기존 테이블 삭제")
        
        # 새 테이블 생성 (컬럼명 변경)
        new_create_sql = create_sql.replace(old_column_name, new_column_name)
        cursor.execute(new_create_sql)
        print(f"  ✓ 새 테이블 생성")
        
        # 모든 컬럼 이름 가져오기
        columns = get_table_columns(cursor, backup_table)
        column_names = [col[1] for col in columns]
        columns_str = ", ".join(column_names)
        
        # 데이터 복사 (컬럼명 매핑)
        new_columns = [new_column_name if c == old_column_name else c for c in column_names]
        new_columns_str = ", ".join(new_columns)
        
        cursor.execute(f"""
            INSERT INTO {table_name} ({new_columns_str})
            SELECT {columns_str} FROM {backup_table}
        """)
        print(f"  ✓ 데이터 복사 완료 ({cursor.rowcount}개 행)")
        
        # 백업 삭제
        cursor.execute(f"DROP TABLE {backup_table}")
        print(f"  ✓ 백업 삭제")
        
        conn.commit()
        print("✅ 컬럼명 변경 완료!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ 에러 발생: {e}")
        
        # 백업에서 복원 시도
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(f"ALTER TABLE {backup_table} RENAME TO {table_name}")
            conn.commit()
            print("  ✓ 백업에서 복원 완료")
        except Exception as restore_error:
            print(f"  ❌ 복원 실패: {restore_error}")
        
        return False


def run_migrations():
    """
    실행할 마이그레이션 목록
    새로운 마이그레이션이 필요할 때 여기에 추가
    """
    print("=" * 60)
    print("데이터베이스 마이그레이션 시작")
    print("=" * 60)
    
    migrations = [
        # 과거 마이그레이션 (참고용)
        # {
        #     "type": "rename",
        #     "old_name": "bm25_hybrid_rank",
        #     "new_name": "bm25_top_rank_in_hybrid",
        #     "description": "BM25 하이브리드 랭크 컬럼명 변경"
        # },
        
        # 새로운 마이그레이션
        {
            "type": "add",
            "column_name": "chat_message_id",
            "column_type": "TEXT",
            "default_value": None,
            "nullable": True,
            "description": "Bot chat message id 컬럼 추가"
        }
    ]
    
    for i, migration in enumerate(migrations, 1):
        print(f"\n[{i}/{len(migrations)}] {migration['description']}")
        print("-" * 60)
        
        if migration["type"] == "add":
            success = add_column(
                column_name=migration["column_name"],
                column_type=migration["column_type"],
                default_value=migration.get("default_value"),
                nullable=migration.get("nullable", True)
            )
        elif migration["type"] == "rename":
            success = rename_column(
                old_column_name=migration["old_name"],
                new_column_name=migration["new_name"]
            )
        else:
            print(f"⚠️ 알 수 없는 마이그레이션 타입: {migration['type']}")
            success = False
        
        if not success:
            print(f"\n⚠️ 마이그레이션 실패. 다음 마이그레이션을 계속 진행합니다.")
    
    print("\n" + "=" * 60)
    print("마이그레이션 완료")
    print("=" * 60)


if __name__ == "__main__":
    run_migrations()
    