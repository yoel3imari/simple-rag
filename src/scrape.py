from __future__ import annotations

import asyncio

import requests

from src.config import JINA_API_KEY, WEB_SCRAPE_TIMEOUT


class ScrapeError(Exception):
    def __init__(self, url: str, message: str):
        self.url = url
        self.message = message
        super().__init__(f"[{url}] {message}")


def _scrape_crawl4ai(
    url: str,
    timeout: int = WEB_SCRAPE_TIMEOUT,
    *,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> str:
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        msg = "crawl4ai is not installed — run `uv sync` to install dependencies"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg)

    async def _crawl():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url, timeout=timeout)
            return result.markdown if result else ""

    try:
        if event_loop is not None:
            markdown = event_loop.run_until_complete(_crawl())
        else:
            markdown = asyncio.run(_crawl())
    except ScrapeError:
        raise
    except Exception as e:
        msg = f"crawl4ai failed: {e}"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg) from e

    if not markdown or not markdown.strip():
        msg = "empty content returned"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg)

    return markdown


def _scrape_jina(url: str, api_key: str | None = None) -> str:
    headers = {"Accept": "text/markdown"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=WEB_SCRAPE_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        msg = "Jina Reader timed out"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg) from None
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        msg = f"Jina Reader returned HTTP {status}"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg) from None
    except requests.exceptions.SSLError:
        msg = "SSL error connecting to Jina Reader"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg) from None
    except requests.exceptions.ConnectionError:
        msg = "connection error to Jina Reader"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg) from None
    except requests.exceptions.RequestException as e:
        msg = f"Jina Reader request failed: {e}"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg) from None

    text = resp.text
    if not text or not text.strip():
        msg = "Jina Reader returned empty content"
        print(f"[scrape] ERROR: {url}: {msg}")
        raise ScrapeError(url, msg)

    return text


def scrape_url(
    url: str,
    *,
    scraper: str = "crawl4ai",
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> str:
    if scraper == "crawl4ai":
        return _scrape_crawl4ai(url, event_loop=event_loop)
    elif scraper == "jina":
        return _scrape_jina(url, api_key=JINA_API_KEY)
    else:
        raise ValueError(f"Unknown scraper: '{scraper}' (expected 'crawl4ai' or 'jina')")
