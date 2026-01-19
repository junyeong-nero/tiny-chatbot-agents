import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from .base import BaseCrawler

# QnA 페이지 카테고리 매핑
# mctgr=11: 자주묻는질문, all: 전체, 01: 인터넷뱅킹, 02: 금융상품, 03: 주식선물옵션, 04: 투자정보, 05: 거래시스템, 07: 기타
QNA_CATEGORIES = {
    "all": "전체",
    "01": "인터넷뱅킹",
    "02": "금융상품",
    "03": "주식선물옵션",
    "04": "투자정보",
    "05": "거래시스템",
    "07": "기타",
}

BASE_URL = "https://www.truefriend.com/main/customer/support/_static/TF04fg000000.jsp"


class QnACrawler(BaseCrawler):
    def __init__(
        self,
        output_dir: str | Path = "data/raw/qna",
        headless: bool = True,
        categories: list[str] | None = None,
    ) -> None:
        super().__init__(output_dir, headless)
        self.categories = categories or list(QNA_CATEGORIES.keys())

    async def crawl(self) -> list[dict[str, Any]]:
        playwright, browser, context, page = await self._create_browser_context()
        all_qna_items = []

        try:
            for category_code in self.categories:
                category_name = QNA_CATEGORIES.get(category_code, category_code)
                self.logger.info(f"Crawling category: {category_name} ({category_code})")

                category_items = await self._crawl_category(page, category_code)
                all_qna_items.extend(category_items)

                self.logger.info(f"Found {len(category_items)} items in {category_name}")
                await asyncio.sleep(1)

        finally:
            await self._close_browser(playwright, browser, context)

        return all_qna_items

    async def _crawl_category(
        self,
        page: Page,
        category_code: str,
    ) -> list[dict[str, Any]]:
        url = f"{BASE_URL}?mctgr={category_code}"
        await page.goto(url)
        await self._wait_for_page_load(page)

        total_pages = await self._get_total_pages(page)
        self.logger.info(f"Total pages: {total_pages}")

        category_items = []

        for page_num in range(1, total_pages + 1):
            if page_num > 1:
                await self._navigate_to_page(page, page_num)

            page_items = await self._extract_qna_items(page, category_code)
            category_items.extend(page_items)

            self.logger.info(f"Page {page_num}: extracted {len(page_items)} items")
            await asyncio.sleep(0.5)

        return category_items

    async def _get_total_pages(self, page: Page) -> int:
        max_page = 1

        all_links = await page.query_selector_all("a")
        for link in all_links:
            onclick = await link.get_attribute("onclick")
            if onclick and "goPage" in onclick:
                match = re.search(r"goPage\(['\"]?(\d+)['\"]?\)", onclick)
                if match:
                    page_num = int(match.group(1))
                    max_page = max(max_page, page_num)

        return max_page

    async def _navigate_to_page(self, page: Page, page_num: int) -> None:
        await page.evaluate(f"goPage('{page_num}')")
        await asyncio.sleep(1)
        await self._wait_for_page_load(page)

    async def _extract_qna_items(
        self,
        page: Page,
        category_code: str,
    ) -> list[dict[str, Any]]:
        items = []

        faq_list = await page.query_selector_all("ul.faq_list > li")

        for idx, li in enumerate(faq_list):
            try:
                question_link = await li.query_selector("a.faq_q")
                if not question_link:
                    continue

                question_text = await question_link.text_content()
                if not question_text:
                    continue

                question_text = self._clean_question_text(question_text)

                await question_link.click()
                await asyncio.sleep(0.3)

                answer_div = await li.query_selector("div.faq_a")
                answer_text = ""
                if answer_div:
                    answer_text = await answer_div.inner_text()
                    answer_text = self._clean_answer_text(answer_text)

                sub_category = self._extract_sub_category(question_text)

                items.append({
                    "question": question_text,
                    "answer": answer_text,
                    "category": QNA_CATEGORIES.get(category_code, category_code),
                    "sub_category": sub_category,
                    "source": "FAQ",
                    "source_url": page.url,
                    "crawled_at": datetime.now().isoformat(),
                })

            except Exception as e:
                self.logger.warning(f"Failed to extract item {idx}: {e}")
                continue

        return items

    def _clean_question_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^질문\s*", "", text)
        return text

    def _clean_answer_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^답변\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_sub_category(self, question: str) -> str:
        match = re.match(r"^\[([^\]]+)\]", question)
        if match:
            return match.group(1)
        return ""


async def main():
    crawler = QnACrawler(headless=True, categories=["all"])
    output_path = await crawler.run()
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
