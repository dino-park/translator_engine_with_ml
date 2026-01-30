"""
ChromaDB 전체 재인덱싱 스크립트

01/26일 잘못된 데이터를 삭제하고 올바른 데이터로 재인덱싱합니다.

사용법:
    python reindex_clean.py
"""

from core.config import load_env, setup_settings
from indexing import run_indexing

def main():
    print("\n" + "="*80)
    print("🗑️  ChromaDB 전체 삭제 및 재인덱싱")
    print("="*80)
    
    # 환경 설정 로드
    env = load_env()
    setup_settings(env)
    
    print("\n⚠️  경고: 기존의 모든 임베딩 데이터가 삭제됩니다!")
    print("   - F:\\chroma_data\\game_translator 의 모든 컬렉션")
    print("   - nodes_cache 파일들")
    print()
    
    response = input("계속하시겠습니까? (yes/no): ")
    
    if response.lower() != "yes":
        print("\n❌ 취소되었습니다.")
        return
    
    print("\n" + "="*80)
    print("📚 올바른 CSV 파일로 재인덱싱 시작")
    print("="*80)
    
    # clear=True로 전체 재인덱싱
    # glossary 폴더의 모든 CSV 파일을 사용합니다
    run_indexing(csv_path=None, clear=True)
    
    print("\n" + "="*80)
    print("✅ 재인덱싱 완료!")
    print("="*80)
    print("\n다음 작업을 권장합니다:")
    print("  1. 서버 재시작 (엔진 캐시 새로고침)")
    print("  2. 테스트 쿼리로 번역 확인")
    print()

if __name__ == "__main__":
    main()


