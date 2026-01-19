#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crawlers import QnACrawler, ToSCrawler


async def crawl_qna(args):
    categories = args.categories.split(",") if args.categories else None
    crawler = QnACrawler(
        output_dir=args.output,
        headless=not args.visible,
        categories=categories,
    )
    return await crawler.run()


async def crawl_tos(args):
    crawler = ToSCrawler(
        output_dir=args.output,
        headless=not args.visible,
    )
    return await crawler.run()


async def crawl_all(args):
    qna_output = await crawl_qna(args)
    print(f"QnA data saved to: {qna_output}")

    args.output = "data/raw/tos"
    tos_output = await crawl_tos(args)
    print(f"ToS data saved to: {tos_output}")


def main():
    parser = argparse.ArgumentParser(description="Crawl QnA and ToS data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    qna_parser = subparsers.add_parser("qna", help="Crawl QnA data")
    qna_parser.add_argument(
        "--output", "-o",
        default="data/raw/qna",
        help="Output directory",
    )
    qna_parser.add_argument(
        "--categories", "-c",
        default=None,
        help="Comma-separated category codes (e.g., all,01,02)",
    )
    qna_parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window",
    )

    tos_parser = subparsers.add_parser("tos", help="Crawl ToS data")
    tos_parser.add_argument(
        "--output", "-o",
        default="data/raw/tos",
        help="Output directory",
    )
    tos_parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window",
    )

    all_parser = subparsers.add_parser("all", help="Crawl both QnA and ToS")
    all_parser.add_argument(
        "--output", "-o",
        default="data/raw/qna",
        help="Output directory for QnA (ToS will use data/raw/tos)",
    )
    all_parser.add_argument(
        "--categories", "-c",
        default=None,
        help="Comma-separated category codes for QnA",
    )
    all_parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window",
    )

    args = parser.parse_args()

    if args.command == "qna":
        asyncio.run(crawl_qna(args))
    elif args.command == "tos":
        asyncio.run(crawl_tos(args))
    elif args.command == "all":
        asyncio.run(crawl_all(args))


if __name__ == "__main__":
    main()
