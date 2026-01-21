"""
Glossary 관리 CLI 도구

사용법:
    python glossary_cli.py list                          # 사용자 추가 항목 조회
    python glossary_cli.py list --type user_term         # user_term만 조회
    python glossary_cli.py delete --cn "综合速配" --ko "종합 매칭"  # 특정 항목 삭제
    python glossary_cli.py delete --type user_term       # 모든 user_term 삭제
"""
import argparse
from engine_core import load_env, setup_settings, list_user_entries, delete_from_glossary_by_metadata


def init():
    """엔진 초기화"""
    env = load_env()
    setup_settings(env)


def cmd_list(args):
    """사용자 추가 항목 조회"""
    init()
    entries = list_user_entries(doc_type=args.type, limit=args.limit)
    
    if not entries:
        print("사용자 추가 항목이 없습니다.")
        return
    
    print(f"{'ID':<45} {'CN':<25} {'KO':<25} {'TYPE'}")
    print("-" * 120)
    for e in entries:
        cn = e['cn'][:22] + "..." if len(e['cn']) > 25 else e['cn']
        ko = e['ko'][:22] + "..." if len(e['ko']) > 25 else e['ko']
        print(f"{e['id']:<45} {cn:<25} {ko:<25} {e['doc_type']}")
    
    print(f"\n총 {len(entries)}개")


def cmd_delete(args):
    """항목 삭제"""
    init()
    
    if not args.cn and not args.ko and not args.type:
        print("삭제 조건을 지정해주세요. (--cn, --ko, --type 중 하나 이상)")
        return
    
    # 삭제 전 확인
    print(f"삭제 조건: cn={args.cn!r}, ko={args.ko!r}, type={args.type!r}")
    
    if not args.yes:
        confirm = input("정말 삭제하시겠습니까? (y/N): ")
        if confirm.lower() != 'y':
            print("취소되었습니다.")
            return
    
    deleted = delete_from_glossary_by_metadata(cn=args.cn, ko=args.ko, doc_type=args.type)
    print(f"삭제 완료: {deleted}개")


def main():
    parser = argparse.ArgumentParser(description="Glossary 관리 CLI")
    subparsers = parser.add_subparsers(dest="command", help="명령어")
    
    # list 명령어
    list_parser = subparsers.add_parser("list", help="사용자 추가 항목 조회")
    list_parser.add_argument("--type", help="문서 유형 필터 (user_term, user_sentence)")
    list_parser.add_argument("--limit", type=int, default=50, help="최대 개수 (기본: 50)")
    
    # delete 명령어
    delete_parser = subparsers.add_parser("delete", help="항목 삭제")
    delete_parser.add_argument("--cn", help="삭제할 중국어 텍스트")
    delete_parser.add_argument("--ko", help="삭제할 한국어 텍스트")
    delete_parser.add_argument("--type", help="삭제할 문서 유형")
    delete_parser.add_argument("-y", "--yes", action="store_true", help="확인 없이 삭제")
    
    args = parser.parse_args()
    
    if args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

