"""
scraper.py - crawl trk.uz and save clean page text to data/pages.json

Install:
    pip install playwright trafilatura beautifulsoup4
    playwright install chromium

Run:
    python scraper.py
"""

import asyncio
import json
import hashlib
from urllib.parse import urljoin, urlparse, urldefrag

import trafilatura
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

START_URL = "https://trk.uz/"
DOMAIN = "trk.uz"
MAX_PAGES = 200            # crawl at most this many pages
MAX_DEPTH = 4             # how deep to follow links
OUTPUT = "data/pages.json"
MIN_TEXT_LEN = 80         # skip near-empty pages

SKIP_EXT = (".jpg", ".png", ".svg", ".css", ".js", ".pdf",
            ".ico", ".webp", ".mp4", ".zip", ".gif")


def clean_url(url: str) -> str:
    url, _ = urldefrag(url)              # drop #fragment
    return url.rstrip("/") or url


def same_site(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == DOMAIN or host.endswith("." + DOMAIN)


async def get_page(page, url: str):
    """Render a page and return its HTML. Retries on failure."""
    for _ in range(3):
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(600)   # let late content load
            return await page.content()
        except Exception:
            await asyncio.sleep(2)
    return None


def extract(html: str):
    """Return (title, text) using trafilatura."""
    data = trafilatura.extract(
        html, output_format="json", with_metadata=True, favor_recall=True
    )
    if not data:
        return "", ""
    d = json.loads(data)
    title = (d.get("title") or "").strip()
    text = " ".join((d.get("text") or "").split())
    return title, text


def find_links(html: str, base: str):
    """Collect internal links on the page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        full = clean_url(urljoin(base, href))
        if same_site(full) and not full.lower().endswith(SKIP_EXT):
            links.add(full)
    return links


async def crawl():
    seen, seen_hashes, pages = set(), set(), []
    current = {clean_url(START_URL)}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        depth = 0
        while current and len(seen) < MAX_PAGES and depth <= MAX_DEPTH:
            next_urls = set()
            for url in sorted(current):
                if url in seen or len(seen) >= MAX_PAGES:
                    continue
                seen.add(url)

                html = await get_page(page, url)
                if not html:
                    print("failed:", url)
                    continue

                title, text = extract(html)
                if len(text) < MIN_TEXT_LEN:
                    continue

                h = hashlib.sha256(text.encode()).hexdigest()
                if h in seen_hashes:            # skip duplicate content
                    continue
                seen_hashes.add(h)

                pages.append({"url": url, "title": title, "content": text})
                print(f"[{len(pages)}] {url}")

                if depth < MAX_DEPTH:
                    next_urls |= find_links(html, url)

            current = next_urls - seen
            depth += 1

        await browser.close()

    return pages


def main():
    pages = asyncio.run(crawl())
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"\nDone: {len(pages)} pages -> {OUTPUT}")


if __name__ == "__main__":
    main()
