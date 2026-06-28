"""
scraper/scraper.py
------------------
Full web scraping pipeline for collecting programming content:
- Recursively crawls documentation sites
- Fetches Stack Overflow Q&A via API
- Fetches GitHub Gists via API
- Fetches dev.to articles via API
- Processes RSS feeds
- Respects robots.txt and rate limits
- Saves raw data as JSONL

Usage:
    python scraper/scraper.py --output data/raw --max-pages 500
    python scraper/scraper.py --source python_docs --output data/raw
    python scraper/scraper.py --config configs/config.yaml
"""

import sys
import os
import re
import time
import json
import hashlib
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
import httpx
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config, clean_url, retry, sanitize_filename
from scraper.robots_check import RobotsChecker

log = get_logger("scraper")
fm = FileManager()


# ============================================================
# Base Scraper
# ============================================================

class BaseScraper:
    """Base class with shared HTTP and rate-limiting logic."""

    def __init__(self, config: Dict):
        self.config = config
        scraper_cfg = config.get("scraper", {})

        self.delay = scraper_cfg.get("request_delay", 1.5)
        self.timeout = scraper_cfg.get("timeout", 30)
        self.max_retries = scraper_cfg.get("max_retries", 3)
        self.user_agent = scraper_cfg.get("user_agent", "Mozilla/5.0")
        self.respect_robots = scraper_cfg.get("respect_robots_txt", True)
        self.output_dir = Path(scraper_cfg.get("output_dir", "data/raw"))

        self.robots = RobotsChecker(user_agent=self.user_agent) if self.respect_robots else None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._last_request_time: Dict[str, float] = {}

    def _rate_limit(self, domain: str) -> None:
        """Enforce per-domain rate limiting."""
        now = time.time()
        last = self._last_request_time.get(domain, 0)
        elapsed = now - last
        wait = self.delay

        if self.robots:
            crawl_delay = self.robots.get_crawl_delay(f"https://{domain}/")
            wait = max(wait, crawl_delay)

        if elapsed < wait:
            time.sleep(wait - elapsed)

        self._last_request_time[domain] = time.time()

    @retry(max_attempts=3, delay=2.0, exceptions=(requests.RequestException,))
    def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[requests.Response]:
        """GET request with rate limiting and robots.txt check."""
        if self.robots and not self.robots.is_allowed(url):
            log.warning(f"Skipping (robots.txt): {url}")
            return None

        domain = urlparse(url).netloc
        self._rate_limit(domain)

        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                log.warning(f"Rate limited on {url}, waiting 60s...")
                time.sleep(60)
                raise
            log.error(f"HTTP error for {url}: {e}")
            return None
        except requests.RequestException as e:
            log.error(f"Request failed for {url}: {e}")
            return None

    def save_record(self, record: Dict, output_file: Path) -> None:
        """Append a record to the JSONL output file."""
        fm.append_jsonl(output_file, record)

    def make_record(self, **kwargs) -> Dict:
        """Create a standardized data record."""
        return {
            "id": hashlib.sha256(kwargs.get("url", "").encode()).hexdigest()[:16],
            "scraped_at": datetime.utcnow().isoformat(),
            **kwargs
        }


# ============================================================
# Recursive Web Scraper (for documentation sites)
# ============================================================

class RecursiveScraper(BaseScraper):
    """
    Crawls documentation sites recursively.
    Respects URL inclusion/exclusion patterns.
    """

    def __init__(self, config: Dict, source: Dict):
        super().__init__(config)
        self.source = source
        self.visited: Set[str] = set()
        self.queue: List[str] = list(source.get("start_urls", []))
        self.allowed_domains = set(source.get("allowed_domains", []))
        self.include_patterns = source.get("url_patterns", {}).get("include", [])
        self.exclude_patterns = source.get("url_patterns", {}).get("exclude", [])
        self.selectors = source.get("selectors", {})
        self.max_pages = source.get("max_pages", 100)

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL matches inclusion/exclusion patterns and domain."""
        parsed = urlparse(url)

        # Domain check
        if self.allowed_domains and parsed.netloc not in self.allowed_domains:
            return False

        # Exclusion check
        for pat in self.exclude_patterns:
            if pat in url:
                return False

        # Inclusion check
        if self.include_patterns:
            for pat in self.include_patterns:
                if pat in url:
                    return True
            return False

        return True

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all valid internal links from a page."""
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full_url = clean_url(urljoin(base_url, href))

            if not full_url.startswith("http"):
                continue
            if full_url in self.visited:
                continue
            if not self._is_allowed_url(full_url):
                continue

            links.append(full_url)
        return links

    def _extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """Extract structured content from a page using configured selectors."""
        content_sel = self.selectors.get("content", "body")
        title_sel = self.selectors.get("title", "h1")
        code_sel = self.selectors.get("code_blocks", "pre code")

        title_tag = soup.select_one(title_sel)
        title = title_tag.get_text(strip=True) if title_tag else ""

        content_tag = soup.select_one(content_sel)
        if not content_tag:
            return None

        # Extract code blocks separately
        code_blocks = []
        for code_tag in content_tag.select(code_sel):
            code_text = code_tag.get_text()
            if len(code_text.strip()) > 20:
                code_blocks.append(code_text.strip())

        # Clean main text
        for tag in content_tag.select("nav, footer, script, style, .sidebar, .nav"):
            tag.decompose()

        full_text = content_tag.get_text(separator="\n", strip=True)

        if len(full_text) < 100:
            return None

        return {
            "title": title,
            "text": full_text,
            "code_blocks": code_blocks,
            "url": url,
        }

    def scrape(self, output_file: Path) -> int:
        """Run the recursive scrape. Returns number of pages scraped."""
        count = 0
        source_name = self.source.get("name", "unknown")

        log.info(f"[{source_name}] Starting recursive scrape. Queue: {len(self.queue)} URLs")

        with tqdm(total=self.max_pages, desc=source_name) as pbar:
            while self.queue and count < self.max_pages:
                url = self.queue.pop(0)
                url = clean_url(url)

                if url in self.visited:
                    continue
                self.visited.add(url)

                response = self.get(url)
                if not response:
                    continue

                try:
                    soup = BeautifulSoup(response.text, "lxml")
                except Exception as e:
                    log.warning(f"Parse error for {url}: {e}")
                    continue

                # Extract content
                content = self._extract_content(soup, url)
                if content:
                    record = self.make_record(
                        source=source_name,
                        content_type=self.source.get("content_type", "documentation"),
                        **content
                    )
                    self.save_record(record, output_file)
                    count += 1
                    pbar.update(1)

                # Find new links
                new_links = self._extract_links(soup, url)
                self.queue.extend(new_links)

        log.info(f"[{source_name}] Scraped {count} pages.")
        return count


# ============================================================
# Stack Overflow API Scraper
# ============================================================

class StackOverflowScraper(BaseScraper):
    """Fetches high-quality Q&A from Stack Overflow API."""

    BASE_URL = "https://api.stackexchange.com/2.3"

    def scrape(self, source: Dict, output_file: Path) -> int:
        count = 0
        max_pages = source.get("max_pages", 10)

        for endpoint in source.get("endpoints", []):
            path = endpoint["path"]
            params = endpoint.get("params", {}).copy()
            params["key"] = os.environ.get("STACKOVERFLOW_KEY", "")

            for page in range(1, max_pages + 1):
                params["page"] = page
                url = self.BASE_URL + path

                response = self.get(url, params=params)
                if not response:
                    break

                data = response.json()
                items = data.get("items", [])

                if not items:
                    break

                for item in tqdm(items, desc=f"SO page {page}", leave=False):
                    question_body = item.get("body", "")
                    title = item.get("title", "")

                    # Clean HTML from body
                    soup = BeautifulSoup(question_body, "lxml")
                    question_text = soup.get_text(separator="\n", strip=True)

                    answers = []
                    if item.get("answers"):
                        for ans in item["answers"][:3]:  # Top 3 answers
                            ans_soup = BeautifulSoup(ans.get("body", ""), "lxml")
                            answers.append({
                                "text": ans_soup.get_text(separator="\n", strip=True),
                                "score": ans.get("score", 0),
                                "is_accepted": ans.get("is_accepted", False),
                            })

                    if len(question_text) < 50:
                        continue

                    record = self.make_record(
                        source="stackoverflow",
                        content_type="qa",
                        url=item.get("link", ""),
                        title=title,
                        text=question_text,
                        answers=answers,
                        tags=item.get("tags", []),
                        score=item.get("score", 0),
                        view_count=item.get("view_count", 0),
                        code_blocks=self._extract_code_from_html(question_body),
                    )
                    self.save_record(record, output_file)
                    count += 1

                # Check if more pages exist
                if not data.get("has_more", False):
                    break
                if data.get("quota_remaining", 1) < 5:
                    log.warning("SO API quota low, stopping.")
                    break

        log.info(f"[stackoverflow] Collected {count} Q&A pairs.")
        return count

    def _extract_code_from_html(self, html: str) -> List[str]:
        """Extract code from HTML code elements."""
        soup = BeautifulSoup(html, "lxml")
        return [code.get_text() for code in soup.find_all("code") if len(code.get_text()) > 10]


# ============================================================
# Dev.to API Scraper
# ============================================================

class DevToScraper(BaseScraper):
    """Fetches developer articles from dev.to API."""

    BASE_URL = "https://dev.to/api"

    def scrape(self, source: Dict, output_file: Path) -> int:
        count = 0
        max_pages = source.get("max_pages", 5)

        for endpoint in source.get("endpoints", []):
            path = endpoint["path"]
            params = endpoint.get("params", {}).copy()

            for page in range(1, max_pages + 1):
                params["page"] = page
                url = self.BASE_URL + path

                response = self.get(url, params=params)
                if not response:
                    break

                articles = response.json()
                if not articles:
                    break

                for article in tqdm(articles, desc=f"devto page {page}", leave=False):
                    # Fetch full article body
                    article_url = f"{self.BASE_URL}/articles/{article['id']}"
                    article_response = self.get(article_url)
                    if not article_response:
                        continue

                    full = article_response.json()
                    body_md = full.get("body_markdown", "")

                    if len(body_md) < 100:
                        continue

                    record = self.make_record(
                        source="devto",
                        content_type="tutorial",
                        url=article.get("url", ""),
                        title=article.get("title", ""),
                        text=body_md,
                        tags=article.get("tag_list", []),
                        code_blocks=self._extract_code_blocks_md(body_md),
                    )
                    self.save_record(record, output_file)
                    count += 1

        log.info(f"[devto] Collected {count} articles.")
        return count

    def _extract_code_blocks_md(self, text: str) -> List[str]:
        """Extract fenced code blocks from markdown."""
        return re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)


# ============================================================
# RSS Feed Scraper
# ============================================================

class RSSFeedScraper(BaseScraper):
    """Scrapes articles linked from RSS feeds."""

    def scrape(self, source: Dict, output_file: Path) -> int:
        import xml.etree.ElementTree as ET

        rss_url = source.get("rss_url")
        max_items = source.get("max_items", 100)
        selectors = source.get("selectors", {})
        source_name = source.get("name", "rss")
        count = 0

        response = self.get(rss_url)
        if not response:
            return 0

        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        items = root.findall(".//item")[:max_items]
        log.info(f"[{source_name}] Found {len(items)} RSS items")

        for item in tqdm(items, desc=source_name):
            link_tag = item.find("link")
            if link_tag is None:
                continue
            url = link_tag.text

            article_response = self.get(url)
            if not article_response:
                continue

            soup = BeautifulSoup(article_response.text, "lxml")

            content_sel = selectors.get("content", "article")
            title_sel = selectors.get("title", "h1")
            code_sel = selectors.get("code_blocks", "pre code")

            title_tag = soup.select_one(title_sel)
            title = title_tag.get_text(strip=True) if title_tag else ""

            content_tag = soup.select_one(content_sel)
            if not content_tag:
                continue

            code_blocks = [c.get_text() for c in content_tag.select(code_sel) if len(c.get_text()) > 10]
            text = content_tag.get_text(separator="\n", strip=True)

            if len(text) < 100:
                continue

            record = self.make_record(
                source=source_name,
                content_type="tutorial",
                url=url,
                title=title,
                text=text,
                code_blocks=code_blocks,
            )
            self.save_record(record, output_file)
            count += 1

        log.info(f"[{source_name}] Collected {count} articles.")
        return count


# ============================================================
# GitHub Gists Scraper
# ============================================================

class GithubGistsScraper(BaseScraper):
    """Fetches public GitHub Gists via the API."""

    BASE_URL = "https://api.github.com"

    def scrape(self, source: Dict, output_file: Path) -> int:
        count = 0
        max_pages = source.get("max_pages", 5)
        allowed_exts = set(source.get("file_extensions", [".py", ".js", ".ts"]))

        headers = {"Accept": "application/vnd.github.v3+json"}
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        for page in range(1, max_pages + 1):
            response = self.get(
                f"{self.BASE_URL}/gists/public",
                params={"per_page": 100, "page": page},
                headers=headers
            )
            if not response:
                break

            gists = response.json()
            if not gists:
                break

            for gist in tqdm(gists, desc=f"GitHub Gists p{page}", leave=False):
                # Filter by file extension
                files = gist.get("files", {})
                relevant_files = {
                    fn: meta for fn, meta in files.items()
                    if any(fn.endswith(ext) for ext in allowed_exts)
                }

                if not relevant_files:
                    continue

                # Fetch full gist content
                gist_response = self.get(f"{self.BASE_URL}/gists/{gist['id']}", headers=headers)
                if not gist_response:
                    continue

                full_gist = gist_response.json()
                code_blocks = []

                for fname, fmeta in full_gist.get("files", {}).items():
                    if any(fname.endswith(ext) for ext in allowed_exts):
                        content = fmeta.get("content", "")
                        if content:
                            code_blocks.append(f"# {fname}\n{content}")

                if not code_blocks:
                    continue

                combined = "\n\n".join(code_blocks)
                record = self.make_record(
                    source="github_gists",
                    content_type="code",
                    url=gist.get("html_url", ""),
                    title=gist.get("description", ""),
                    text=combined,
                    code_blocks=code_blocks,
                    files=list(relevant_files.keys()),
                )
                self.save_record(record, output_file)
                count += 1

        log.info(f"[github_gists] Collected {count} gists.")
        return count


# ============================================================
# Pipeline Orchestrator
# ============================================================

class ScraperPipeline:
    """
    Orchestrates all scrapers based on sources.yaml config.
    Runs each enabled source and saves to JSONL.
    """

    def __init__(self, config: Dict, sources: List[Dict]):
        self.config = config
        self.sources = sources
        self.output_dir = Path(config.get("scraper", {}).get("output_dir", "data/raw"))
        fm._ensure_dir(self.output_dir)

    def run(self, source_filter: Optional[str] = None) -> Dict[str, int]:
        """
        Run all enabled scrapers.

        Args:
            source_filter: If provided, only run this source name.

        Returns:
            Dict of source_name -> pages_scraped
        """
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for source in self.sources:
            name = source.get("name", "unknown")
            enabled = source.get("enabled", True)

            if source_filter and name != source_filter:
                continue
            if not enabled:
                log.info(f"Skipping disabled source: {name}")
                continue

            output_file = self.output_dir / f"{name}_{timestamp}.jsonl"
            log.info(f"{'='*50}")
            log.info(f"Starting source: {name} ({source.get('type', '?')})")
            log.info(f"Output: {output_file}")

            try:
                count = self._run_source(source, output_file)
                results[name] = count
                log.info(f"Completed {name}: {count} records")
            except Exception as e:
                log.error(f"Source {name} failed: {e}", exc_info=True)
                results[name] = 0

        log.info(f"\n{'='*50}")
        log.info("SCRAPING COMPLETE")
        for name, count in results.items():
            log.info(f"  {name}: {count} records")
        log.info(f"Total: {sum(results.values())} records")
        log.info(f"Output directory: {self.output_dir}")

        return results

    def _run_source(self, source: Dict, output_file: Path) -> int:
        """Run appropriate scraper for a source type."""
        source_type = source.get("type")

        if source_type == "recursive":
            scraper = RecursiveScraper(self.config, source)
            return scraper.scrape(output_file)

        elif source_type == "api":
            name = source.get("name")
            if name == "stackoverflow":
                scraper = StackOverflowScraper(self.config)
                return scraper.scrape(source, output_file)
            elif name == "devto":
                scraper = DevToScraper(self.config)
                return scraper.scrape(source, output_file)
            elif name == "github_gists":
                scraper = GithubGistsScraper(self.config)
                return scraper.scrape(source, output_file)
            else:
                log.warning(f"Unknown API source: {name}")
                return 0

        elif source_type == "rss":
            scraper = RSSFeedScraper(self.config)
            return scraper.scrape(source, output_file)

        else:
            log.warning(f"Unknown source type: {source_type}")
            return 0


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Web scraping pipeline for coding content")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--sources", default="scraper/sources.yaml", help="Sources config file")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--source", default=None, help="Run only this source name")
    parser.add_argument("--max-pages", type=int, default=None, help="Override max pages per source")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configs
    config = load_config(args.config)
    with open(args.sources, "r") as f:
        sources_config = yaml.safe_load(f)

    sources = sources_config.get("sources", [])

    # Apply CLI overrides
    if args.output:
        config.setdefault("scraper", {})["output_dir"] = args.output
    if args.max_pages:
        for source in sources:
            source["max_pages"] = min(source.get("max_pages", args.max_pages), args.max_pages)

    # Run pipeline
    pipeline = ScraperPipeline(config, sources)
    results = pipeline.run(source_filter=args.source)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
