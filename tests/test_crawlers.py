import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crawlers import QnACrawler, ToSCrawler


@pytest.fixture
def qna_crawler(tmp_path):
    return QnACrawler(output_dir=tmp_path / "qna", headless=True, categories=["all"])


@pytest.fixture
def tos_crawler(tmp_path):
    return ToSCrawler(output_dir=tmp_path / "tos", headless=True)


class TestQnACrawler:
    @pytest.mark.asyncio
    async def test_crawl_single_page(self, qna_crawler):
        """QnA 크롤러가 최소 1개 이상의 항목을 가져오는지 테스트"""
        playwright, browser, context, page = await qna_crawler._create_browser_context()

        try:
            await page.goto(
                "https://www.truefriend.com/main/customer/support/_static/TF04fg000000.jsp?mctgr=all"
            )
            await qna_crawler._wait_for_page_load(page)

            items = await qna_crawler._extract_qna_items(page, "all")

            assert len(items) > 0, "No QnA items extracted"

            first_item = items[0]
            assert "question" in first_item
            assert "answer" in first_item
            assert "category" in first_item
            assert first_item["question"], "Question should not be empty"

        finally:
            await qna_crawler._close_browser(playwright, browser, context)

    @pytest.mark.asyncio
    async def test_clean_question_text(self, qna_crawler):
        """질문 텍스트 정제 테스트"""
        raw = "질문 [금현물] 금현물 거래는 어떻게 하나요?"
        cleaned = qna_crawler._clean_question_text(raw)

        assert not cleaned.startswith("질문")
        assert "[금현물]" in cleaned

    @pytest.mark.asyncio
    async def test_extract_sub_category(self, qna_crawler):
        """서브카테고리 추출 테스트"""
        question = "[금현물] 금현물 거래는 어떻게 하나요?"
        sub_cat = qna_crawler._extract_sub_category(question)

        assert sub_cat == "금현물"

        question_no_cat = "일반 질문입니다"
        sub_cat_empty = qna_crawler._extract_sub_category(question_no_cat)

        assert sub_cat_empty == ""


class TestToSCrawler:
    @pytest.mark.asyncio
    async def test_crawl_tos_list(self, tos_crawler):
        """ToS 크롤러가 목록을 가져오는지 테스트"""
        playwright, browser, context, page = await tos_crawler._create_browser_context()

        try:
            await page.goto("https://www.truefriend.com/main/customer/guide/Guide.jsp")
            await tos_crawler._wait_for_page_load(page)

            items = await tos_crawler._extract_tos_list(page, 1)

            assert len(items) > 0, "No ToS items extracted"

            first_item = items[0]
            assert "title" in first_item
            assert "category" in first_item
            assert "num" in first_item

        finally:
            await tos_crawler._close_browser(playwright, browser, context)

    @pytest.mark.asyncio
    async def test_parse_doview_params(self, tos_crawler):
        """doView 파라미터 파싱 테스트"""
        onclick = "doView('78','1');"
        num, row = tos_crawler._parse_doview_params(onclick)

        assert num == "78"
        assert row == "1"

    @pytest.mark.asyncio
    async def test_extract_effective_date(self, tos_crawler):
        """시행일 추출 테스트"""
        text = "이 약관은 2024년 01월 15일부터 시행합니다."
        date = tos_crawler._extract_effective_date(text)

        assert date == "2024-01-15"

    @pytest.mark.asyncio
    async def test_extract_revision_history(self, tos_crawler):
        """개정 이력 추출 테스트"""
        text = "제정 2009. 06. 12. 개정 2014. 10. 13 개정 2020. 12. 10"
        revisions = tos_crawler._extract_revision_history(text)

        assert len(revisions) >= 2
        assert "2009-06-12" in revisions


class TestIntegration:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_qna_full_crawl_limited(self, tmp_path):
        """QnA 전체 크롤링 테스트 (첫 페이지만)"""
        crawler = QnACrawler(
            output_dir=tmp_path / "qna",
            headless=True,
            categories=["all"],
        )

        playwright, browser, context, page = await crawler._create_browser_context()

        try:
            items = await crawler._crawl_category(page, "all")
            assert len(items) > 0

            output_path = await crawler.save_json(items, "test_qna.json")
            assert output_path.exists()

            with open(output_path, encoding="utf-8") as f:
                saved_data = json.load(f)
            assert len(saved_data) == len(items)

        finally:
            await crawler._close_browser(playwright, browser, context)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
