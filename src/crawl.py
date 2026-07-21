from __future__ import annotations

import asyncio
import random
import re
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone

from src.config import (
    WEB_CRAWL_DELAY,
    WEB_CRAWL_MAX_DEPTH,
    WEB_CRAWL_MAX_PAGES,
    WEB_SCRAPER_BACKEND,
)
from src.scrape import ScrapeError, scrape_url

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def _normalize_url(raw: str, base: str | None = None) -> str | None:
    parsed = urllib.parse.urlparse(raw)
    # Only HTTP(S)
    if parsed.scheme not in ("http", "https", ""):
        return None
    # Resolve relative
    if not parsed.netloc and base is not None:
        raw = urllib.parse.urljoin(base, raw)
        parsed = urllib.parse.urlparse(raw)
    hostname = parsed.hostname
    if not parsed.netloc or not hostname:
        return None
    # Normalise: lowercase scheme+host+path, strip fragment, strip trailing slash
    scheme = parsed.scheme.lower()
    host = hostname.lower()
    if parsed.port and (
        (scheme == "http" and parsed.port == 80)
        or (scheme == "https" and parsed.port == 443)
    ):
        host = f"{host}:{parsed.port}"
    path = parsed.path.lower().rstrip("/") or "/"
    return f"{scheme}://{host}{path}"


def _extract_links(markdown: str, page_url: str) -> list[str]:
    links: list[str] = []
    for _, href in _LINK_RE.findall(markdown):
        normalized = _normalize_url(href, page_url)
        if normalized:
            links.append(normalized)
    return links


def _extract_title(markdown: str, url: str) -> str:
    m = _H1_RE.search(markdown)
    if m:
        return m.group(1).strip()
    return urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1] or url


def crawl_site(
    start_url: str,
    *,
    max_pages: int = WEB_CRAWL_MAX_PAGES,
    max_depth: int = WEB_CRAWL_MAX_DEPTH,
    delay: float = WEB_CRAWL_DELAY,
    same_domain: bool = True,
    respect_robots: bool = True,
    scraper: str = WEB_SCRAPER_BACKEND,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> list[dict]:
    start_norm = _normalize_url(start_url)
    if start_norm is None:
        raise ValueError(f"Invalid start URL: {start_url}")

    start_netloc = urllib.parse.urlparse(start_norm).netloc

    # robots.txt
    rp = urllib.robotparser.RobotFileParser()
    if respect_robots:
        rp.set_url(f"{urllib.parse.urlunparse(urllib.parse.urlparse(start_norm)._replace(path='/robots.txt'))}")
        try:
            rp.read()
        except Exception:
            pass  # if robots.txt is unreachable, proceed without it

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start_norm, 0)]
    pages: list[dict] = []

    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)

        if url in visited:
            continue

        if respect_robots and not rp.can_fetch("*", url):
            print(f"[crawl] Skipping (robots.txt): {url}")
            visited.add(url)
            continue

        visited.add(url)

        print(f"[crawl] Fetching ({len(pages) + 1}/{max_pages}) depth={depth} {url}")

        try:
            markdown = scrape_url(url, scraper=scraper, event_loop=event_loop)
        except ScrapeError as e:
            print(f"[crawl] WARNING: {e.message}")
            continue

        fetch_time = datetime.now(timezone.utc).isoformat()
        title = _extract_title(markdown, url)

        pages.append({
            "url": url,
            "title": title,
            "markdown": markdown,
            "depth": depth,
            "fetch_time": fetch_time,
        })

        # Enqueue child links if within depth limit
        if depth < max_depth:
            links = _extract_links(markdown, url)
            for link in links:
                if not same_domain or urllib.parse.urlparse(link).netloc == start_netloc:
                    if link not in visited:
                        queue.append((link, depth + 1))

        # Rate limiting
        if queue and len(pages) < max_pages:
            jitter = random.uniform(0, delay * 0.5)
            time.sleep(delay + jitter)

    return pages
