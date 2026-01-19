import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class BaseCrawler(ABC):
    def __init__(
        self,
        output_dir: str | Path,
        headless: bool = True,
        timeout: int = 30000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def crawl(self) -> list[dict[str, Any]]:
        pass

    async def save_json(
        self,
        data: list[dict[str, Any]],
        filename: str | None = None,
    ) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__.lower()}_{timestamp}.json"

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved {len(data)} items to {output_path}")
        return output_path

    async def run(self) -> Path:
        self.logger.info(f"Starting {self.__class__.__name__}")
        data = await self.crawl()
        output_path = await self.save_json(data)
        self.logger.info(f"Completed. Total items: {len(data)}")
        return output_path

    async def _create_browser_context(self):
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=self.headless)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        page.set_default_timeout(self.timeout)
        return playwright, browser, context, page

    async def _close_browser(self, playwright, browser, context) -> None:
        await context.close()
        await browser.close()
        await playwright.stop()

    async def _wait_for_page_load(self, page: Page) -> None:
        await page.wait_for_load_state("networkidle")

    async def _safe_get_text(
        self,
        page: Page,
        selector: str,
        default: str = "",
    ) -> str:
        try:
            element = await page.query_selector(selector)
            if element:
                text = await element.text_content()
                return text.strip() if text else default
        except Exception as e:
            self.logger.debug(f"Failed to get text for {selector}: {e}")
        return default
