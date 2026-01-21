"""
사용자 추가 데이터(user_term, user_sentence) 관리 스크립트

사용법:
    # ===== 목록 조회 =====
    python cleanup_user_terms.py list                    # 전체 목록 (term + sentence)
    python cleanup_user_terms.py list term               # user_term만
    python cleanup_user_terms.py list sentence           # user_sentence만
    
    # ===== 전체 삭제 =====
    python cleanup_user_terms.py delete_all              # 전체 삭제 (term + sentence)
    python cleanup_user_terms.py delete_all term         # 모든 user_term 삭제
    python cleanup_user_terms.py delete_all sentence     # 모든 user_sentence 삭제
    
    # ===== 특정 항목 삭제 =====
    python cleanup_user_terms.py delete --cn "텍스트"    # 특정 cn으로 삭제
    python cleanup_user_terms.py delete --ko "텍스트"    # 특정 ko로 삭제
    python cleanup_user_terms.py delete --id "doc_id"    # 특정 ID로 삭제
    
    # ===== 통계 =====
    python cleanup_user_terms.py stats                   # 통계 출력
"""
import sys

from core.config import load_env, setup_settings
from core.chroma_utils import get_chroma_collection
from core.chroma_utils import delete_from_glossary_by_metadata

from llama_index.core import Settings


# ===== 목록 조회 =====

def list_user_data(doc_type: str = None):
    """사용자 데이터 목록 출력"""
    collection = get_chroma_collection(Settings.env["PERSIST_DIR"])
    
    if doc_type == "term":
        where = {"doc_type": "user_term"}
        title = "user_term"
    elif doc_type == "sentence":
        where = {"doc_type": "user_sentence"}
        title = "user_sentence"
    else:
        where = {"$or": [{"doc_type": "user_term"}, {"doc_type": "user_sentence"}]}
        title = "user_term + user_sentence"
    
    results = collection.get(where=where)
    
    print(f"\n=== {title} 목록 ({len(results['ids'])}개) ===\n")
    
    for i, (doc_id, meta) in enumerate(zip(results["ids"], results["metadatas"])):
        dtype = meta.get("doc_type", "unknown")
        cn = meta.get("cn", "")
        ko = meta.get("ko", "")
        created = meta.get("created_at", "unknown")
        
        # 긴 텍스트 줄임
        cn_display = cn[:60] + "..." if len(cn) > 60 else cn
        ko_display = ko[:60] + "..." if len(ko) > 60 else ko
        
        type_label = "[TERM]" if dtype == "user_term" else "[SENT]"
        
        print(f"[{i+1}] {type_label} {created}")
        print(f"    ID: {doc_id}")
        print(f"    cn: {cn_display}")
        print(f"    ko: {ko_display}")
        print()


# ===== 삭제 =====

def delete_all_user_data(doc_type: str = None):
    """사용자 데이터 전체 삭제"""
    if doc_type == "term":
        count = delete_from_glossary_by_metadata(doc_type="user_term")
        print(f"\n✅ {count}개의 user_term이 삭제되었습니다.\n")
    elif doc_type == "sentence":
        count = delete_from_glossary_by_metadata(doc_type="user_sentence")
        print(f"\n✅ {count}개의 user_sentence가 삭제되었습니다.\n")
    else:
        count_term = delete_from_glossary_by_metadata(doc_type="user_term")
        count_sentence = delete_from_glossary_by_metadata(doc_type="user_sentence")
        print(f"\n✅ user_term {count_term}개, user_sentence {count_sentence}개가 삭제되었습니다.\n")


def delete_by_cn(cn_text: str):
    """특정 cn으로 삭제"""
    count = delete_from_glossary_by_metadata(cn=cn_text)
    print(f"\n✅ cn='{cn_text}' 항목 {count}개가 삭제되었습니다.\n")


def delete_by_ko(ko_text: str):
    """특정 ko로 삭제"""
    count = delete_from_glossary_by_metadata(ko=ko_text)
    print(f"\n✅ ko='{ko_text}' 항목 {count}개가 삭제되었습니다.\n")


def delete_by_id(doc_id: str):
    """특정 ID로 삭제"""
    try:
        collection = get_chroma_collection(Settings.env["PERSIST_DIR"])
        collection.delete(ids=[doc_id])
        print(f"\n✅ ID='{doc_id}' 항목이 삭제되었습니다.\n")
    except Exception as e:
        print(f"\n❌ 삭제 실패: {e}\n")


# ===== 통계 =====

def show_stats():
    """통계 출력"""
    collection = get_chroma_collection(Settings.env["PERSIST_DIR"])
    
    # user_term 개수
    term_results = collection.get(where={"doc_type": "user_term"})
    term_count = len(term_results["ids"]) if term_results["ids"] else 0
    
    # user_sentence 개수
    sentence_results = collection.get(where={"doc_type": "user_sentence"})
    sentence_count = len(sentence_results["ids"]) if sentence_results["ids"] else 0
    
    print(f"\n{'='*40}")
    print("      📊 사용자 데이터 통계")
    print(f"{'='*40}")
    print(f"  user_term     : {term_count:>5}개")
    print(f"  user_sentence : {sentence_count:>5}개")
    print(f"{'─'*40}")
    print(f"  합계          : {term_count + sentence_count:>5}개")
    print(f"{'='*40}\n")


# ===== 메인 =====

def print_help():
    print(__doc__)


if __name__ == "__main__":
    # 환경 설정
    env = load_env()
    setup_settings(env)
    
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    # list 명령
    if cmd == "list":
        doc_type = sys.argv[2] if len(sys.argv) >= 3 else None
        list_user_data(doc_type)
    
    # delete_all 명령
    elif cmd == "delete_all":
        doc_type = sys.argv[2] if len(sys.argv) >= 3 else None
        
        if doc_type == "term":
            msg = "정말 모든 user_term을 삭제하시겠습니까?"
        elif doc_type == "sentence":
            msg = "정말 모든 user_sentence를 삭제하시겠습니까?"
        else:
            msg = "정말 모든 user_term과 user_sentence를 삭제하시겠습니까?"
        
        confirm = input(f"{msg} (y/N): ")
        if confirm.lower() == "y":
            delete_all_user_data(doc_type)
        else:
            print("취소되었습니다.")
    
    # delete 명령
    elif cmd == "delete":
        if len(sys.argv) >= 4:
            option = sys.argv[2]
            value = sys.argv[3]
            
            if option == "--cn":
                delete_by_cn(value)
            elif option == "--ko":
                delete_by_ko(value)
            elif option == "--id":
                delete_by_id(value)
            else:
                print_help()
        else:
            print_help()
    
    # stats 명령
    elif cmd == "stats":
        show_stats()
    
    else:
        print_help()
