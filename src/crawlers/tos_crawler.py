import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from .base import BaseCrawler

# 약관 카테고리
# type=1: 전체, type=2: 개인(신용)정보관련 사항
TOS_TYPES = {
    "1": "전체",
    "2": "개인(신용)정보관련사항",
}

# 약관 구분 (서비스약관, 설명서, 고객유의사항 등)
TOS_CATEGORIES = {
    "서비스약관": "서비스약관",
    "설명서": "설명서",
    "고객유의사항": "고객유의사항",
}

BASE_URL = "https://www.truefriend.com/main/customer/guide/Guide.jsp"
DETAIL_URL_TEMPLATE = (
    "https://www.truefriend.com/main/customer/guide/Guide.jsp"
    "?&cmd=TF04ag010002&currentPage={page}&num={num}&rowNum={row}"
)


class ToSCrawler(BaseCrawler):
    def __init__(
        self,
        output_dir: str | Path = "data/raw/tos",
        headless: bool = True,
        include_pdf: bool = True,
    ) -> None:
        super().__init__(output_dir, headless)
        self.include_pdf = include_pdf

    async def crawl(self) -> list[dict[str, Any]]:
        playwright, browser, context, page = await self._create_browser_context()
        all_tos_items = []

        try:
            await page.goto(BASE_URL)
            await self._wait_for_page_load(page)

            total_pages = await self._get_total_pages(page)
            self.logger.info(f"Total pages: {total_pages}")

            for page_num in range(1, total_pages + 1):
                if page_num > 1:
                    await self._navigate_to_page(page, page_num)

                page_items = await self._extract_tos_list(page, page_num)
                self.logger.info(f"Page {page_num}: found {len(page_items)} items")

                for item in page_items:
                    detail = await self._extract_tos_detail(page, item)
                    if detail:
                        all_tos_items.append(detail)

                    await asyncio.sleep(0.5)

                await asyncio.sleep(1)

        finally:
            await self._close_browser(playwright, browser, context)

        return all_tos_items

    async def _get_total_pages(self, page: Page) -> int:
        pagination_links = await page.query_selector_all(
            "div.paginate a[href*='javascript:void(0)']"
        )

        max_page = 1
        for link in pagination_links:
            text = await link.text_content()
            if text and text.strip().isdigit():
                max_page = max(max_page, int(text.strip()))

        last_btn = await page.query_selector("a.btn_last")
        if last_btn:
            onclick = await last_btn.get_attribute("onclick")
            if onclick:
                match = re.search(r"goPage\((\d+)\)", onclick)
                if match:
                    max_page = max(max_page, int(match.group(1)))

        return max_page

    async def _navigate_to_page(self, page: Page, page_num: int) -> None:
        await page.evaluate(f"goPage({page_num})")
        await self._wait_for_page_load(page)

    async def _extract_tos_list(
        self,
        page: Page,
        current_page: int,
    ) -> list[dict[str, Any]]:
        items = []

        rows = await page.query_selector_all("table tbody tr")

        for row_idx, row in enumerate(rows, 1):
            try:
                cells = await row.query_selector_all("td")
                if len(cells) < 2:
                    continue

                category_cell = cells[0]
                title_cell = cells[1]

                category = await category_cell.text_content()
                category = category.strip() if category else ""

                title_link = await title_cell.query_selector("a")
                if not title_link:
                    continue

                title = await title_link.text_content()
                title = title.strip() if title else ""

                onclick = await title_link.get_attribute("onclick")
                num, row_num = self._parse_doview_params(onclick)

                pdf_url = None
                if len(cells) > 2:
                    pdf_link = await cells[2].query_selector("a")
                    if pdf_link:
                        pdf_onclick = await pdf_link.get_attribute("onclick")
                        pdf_url = self._parse_dofile_url(pdf_onclick)

                items.append({
                    "category": category,
                    "title": title,
                    "num": num,
                    "row_num": row_num,
                    "current_page": current_page,
                    "pdf_url": pdf_url,
                })

            except Exception as e:
                self.logger.warning(f"Failed to extract row {row_idx}: {e}")
                continue

        return items

    def _parse_doview_params(self, onclick: str | None) -> tuple[str, str]:
        if not onclick:
            return "", ""

        match = re.search(r"doView\(['\"]?(\d+)['\"]?\s*,\s*['\"]?(\d+)['\"]?\)", onclick)
        if match:
            return match.group(1), match.group(2)
        return "", ""

    def _parse_dofile_url(self, onclick: str | None) -> str | None:
        if not onclick:
            return None

        match = re.search(r"doFile\(['\"]([^'\"]+)['\"]\)", onclick)
        if match:
            return match.group(1)
        return None

    async def _extract_tos_detail(
        self,
        page: Page,
        item: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not item["num"]:
            return None

        detail_url = DETAIL_URL_TEMPLATE.format(
            page=item["current_page"],
            num=item["num"],
            row=item["row_num"],
        )

        try:
            await page.goto(detail_url)
            await self._wait_for_page_load(page)

            content_iframe = await page.query_selector("iframe")
            content_text = ""
            sections = []

            if content_iframe:
                frame = await content_iframe.content_frame()
                if frame:
                    content_text = await frame.inner_text("body")
                    sections = await self._extract_sections(frame)

            effective_date = self._extract_effective_date(content_text)
            revision_history = self._extract_revision_history(content_text)

            return {
                "title": item["title"],
                "category": item["category"],
                "content": content_text.strip(),
                "sections": sections,
                "effective_date": effective_date,
                "revision_history": revision_history,
                "pdf_url": item["pdf_url"],
                "source_url": detail_url,
                "source": "ToS",
                "crawled_at": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.warning(f"Failed to extract detail for {item['title']}: {e}")
            return None

    async def _extract_sections(self, frame) -> list[dict[str, str]]:
        sections = []

        headings = await frame.query_selector_all("h2")

        for heading in headings:
            section_title = await heading.text_content()
            if not section_title:
                continue

            section_title = section_title.strip()

            next_sibling = await heading.evaluate_handle(
                "(el) => el.nextElementSibling"
            )
            section_content = ""

            if next_sibling:
                tag_name = await next_sibling.evaluate("(el) => el.tagName")
                if tag_name and tag_name.lower() in ["p", "ul", "ol", "div"]:
                    section_content = await next_sibling.inner_text()

            sections.append({
                "title": section_title,
                "content": section_content.strip() if section_content else "",
            })

        return sections

    def _extract_effective_date(self, text: str) -> str:
        patterns = [
            r"(\d{4})\s*[년.]\s*(\d{1,2})\s*[월.]\s*(\d{1,2})\s*일?\s*부터\s*시행",
            r"시행일\s*[:：]?\s*(\d{4})[./년](\d{1,2})[./월](\d{1,2})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                year, month, day = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        return ""

    def _extract_revision_history(self, text: str) -> list[str]:
        revisions = []

        pattern = r"(?:개정|제정)\s*(\d{4})\s*[./년]\s*(\d{1,2})\s*[./월]\s*(\d{1,2})"
        matches = re.findall(pattern, text)

        for year, month, day in matches:
            revisions.append(f"{year}-{month.zfill(2)}-{day.zfill(2)}")

        return sorted(set(revisions))


async def main():
    crawler = ToSCrawler(headless=True)
    output_path = await crawler.run()
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
