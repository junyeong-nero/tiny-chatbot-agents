#!/usr/bin/env python3
"""Backfill human agent (상담원) answers into QnA vector store.

This script implements the "상담원 답변 자동 추가" feature described in README,
enabling continuous learning from live customer support interactions.

Usage:
    python scripts/backfill_agent_answers.py --file data/raw/agent_answers.json
    python scripts/backfill_agent_answers.py --file answers.json --check-duplicates
    python scripts/backfill_agent_answers.py --search "계좌 해지"
"""

import argparse
import importlib
import json
import sys
from pathlib import Path


def load_backfill():
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    module = importlib.import_module("src.vectorstore")
    return module.HumanAgentBackfill


def main() -> None:
    parser = argparse.ArgumentParser(description="상담원 답변을 QnA DB에 추가합니다")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="상담원 답변 JSON 파일 경로",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="data/vectordb/qna",
        help="ChromaDB 저장 디렉토리",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="embedding_config.yaml의 모델 키",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="중복 질문 확인 후 추가 (임계값: 0.90)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.90,
        help="중복 판정 유사도 임계값 (기본값: 0.90)",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        help="기존 답변 검색 테스트",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="검색 결과 개수",
    )
    parser.add_argument(
        "--add",
        nargs=2,
        metavar=("QUESTION", "ANSWER"),
        help="단일 질문-답변 쌍 추가 (예: --add '질문' '답변')",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="--add 사용 시 카테고리",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="저장소 통계 출력",
    )

    args = parser.parse_args()

    print("상담원 답변 Backfill 초기화 중...")
    print(f"  - 모델: {args.model or '기본값'}")
    print(f"  - DB 디렉토리: {args.db_dir}")

    HumanAgentBackfill = load_backfill()
    backfill = HumanAgentBackfill(
        persist_directory=args.db_dir,
        embedding_model=args.model,
    )

    print(f"현재 문서 수: {backfill.qna_store.count()}")

    if args.stats:
        stats = backfill.get_stats()
        print("\n저장소 통계:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        return

    if args.search:
        print(f"\n🔍 검색: '{args.search}'")
        results = backfill.search_existing(
            args.search,
            n_results=args.top_k,
            score_threshold=0.0,
        )

        if results:
            print(f"   {len(results)}개 결과:\n")
            for i, r in enumerate(results, 1):
                print(f"   [{i}] Score: {r['score']:.4f} | Source: {r['source']}")
                print(f"       Q: {r['question'][:80]}...")
                print(f"       A: {r['answer'][:80]}...")
                print()
        else:
            print("   결과 없음.")
        return

    if args.add:
        question, answer = args.add
        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            print("오류: 질문/답변이 비어있습니다.")
            return
        print("\n단일 답변 추가:")
        print(f"   질문: {question}")
        print(f"   답변: {answer[:50]}...")

        try:
            if args.check_duplicates:
                qna_id, is_dup = backfill.add_if_not_duplicate(
                    question=question,
                    answer=answer,
                    category=args.category,
                    duplicate_threshold=args.duplicate_threshold,
                )
                if is_dup:
                    print("   ⚠️  중복 질문 발견 - 추가하지 않았습니다.")
                else:
                    print(f"   ✅ 추가 완료: {qna_id}")
            else:
                qna_id = backfill.add_agent_answer(
                    question=question,
                    answer=answer,
                    category=args.category,
                )
                print(f"   ✅ 추가 완료: {qna_id}")
        except ValueError as e:
            print(f"오류: {e}")
            return

        print(f"   총 문서 수: {backfill.qna_store.count()}")
        return

    if args.file:
        file_path = Path(args.file)
        print(f"\n📁 파일에서 로드: {file_path}")

        if args.check_duplicates:
            if not file_path.exists():
                print(f"파일을 찾을 수 없습니다: {file_path}")
                return

            try:
                items = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"JSON 파싱 오류: {e}")
                return

            if not isinstance(items, list):
                print("JSON 형식 오류: 배열이 아닙니다.")
                return

            added = 0
            skipped_dup = 0
            skipped_empty = 0

            for item in items:
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()

                if not question or not answer:
                    skipped_empty += 1
                    continue

                qna_id, is_dup = backfill.add_if_not_duplicate(
                    question=question,
                    answer=answer,
                    category=item.get("category", ""),
                    sub_category=item.get("sub_category", ""),
                    duplicate_threshold=args.duplicate_threshold,
                )
                if is_dup:
                    skipped_dup += 1
                else:
                    added += 1

            print("\nBackfill 완료!")
            print(f"   추가: {added}")
            print(f"   중복 건너뜀: {skipped_dup}")
            print(f"   빈 항목 건너뜀: {skipped_empty}")
        else:
            result = backfill.load_from_json(file_path)
            print("\nBackfill 완료!")
            print(f"   추가: {result.added_count}")
            print(f"   건너뜀: {result.skipped_count}")
            if result.errors:
                print(f"   오류: {len(result.errors)}")
                for err in result.errors[:5]:
                    print(f"      - {err}")

        print(f"   총 문서 수: {backfill.qna_store.count()}")
        return

    parser.print_help()
    print("\n💡 예시:")
    print("   python scripts/backfill_agent_answers.py --file data/raw/agent_answers.json")
    print("   python scripts/backfill_agent_answers.py --add '질문' '답변' --category '계좌'")
    print("   python scripts/backfill_agent_answers.py --search '계좌 해지'")


if __name__ == "__main__":
    main()
