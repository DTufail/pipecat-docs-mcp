"""
Main scraper: fetches sitemap, scrapes each page, writes chunks.jsonl.

Usage:
    python scraper.py --output data/chunks.jsonl
    python scraper.py --resume
    python scraper.py --sections getting-started,guides
    python scraper.py --debug --save-html
    python scraper.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm

from config import (
    BASE_URL, SITEMAP_URL, HEADERS,
    RATE_LIMIT_DELAY, MAX_RETRIES, BACKOFF_FACTOR, RETRY_STATUS_CODES,
    SAVE_EVERY_N_PAGES, KNOWN_SECTIONS, SECTION_MAP, SELECTORS,
    OUTPUT_FILE, RAW_HTML_DIR, ERROR_LOG_FILE, PROGRESS_FILE,
)
from utils import (
    is_doc_url, url_to_section, parse_page_to_chunks,
    write_chunks_jsonl, load_scraped_urls,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ERROR_LOG_FILE, encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_with_retry(
    session: requests.Session,
    url: str,
    max_retries: int = MAX_RETRIES,
) -> Optional[requests.Response]:
    """
    GET a URL with exponential backoff on transient errors.

    Returns the Response on success, None on permanent failure.
    """
    delay = RATE_LIMIT_DELAY
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (404, 410):
                logger.warning("Skip %s — HTTP %s", url, resp.status_code)
                return None
            if resp.status_code in RETRY_STATUS_CODES:
                logger.warning(
                    "HTTP %s for %s — retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, url, delay, attempt, max_retries,
                )
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
                continue
            logger.error("Unexpected HTTP %s for %s", resp.status_code, url)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error for %s: %s (attempt %d/%d)", url, exc, attempt, max_retries)
            time.sleep(delay)
            delay *= BACKOFF_FACTOR

    logger.error("Giving up on %s after %d attempts", url, max_retries)
    return None


# ---------------------------------------------------------------------------
# Sitemap
# ---------------------------------------------------------------------------

def fetch_sitemap_urls(session: requests.Session) -> List[str]:
    """
    Fetch and parse sitemap.xml, returning all doc page URLs.
    Handles both flat sitemaps and sitemap index files.
    """
    logger.info("Fetching sitemap: %s", SITEMAP_URL)
    resp = fetch_with_retry(session, SITEMAP_URL)
    if resp is None:
        logger.error("Could not fetch sitemap — aborting.")
        sys.exit(1)

    urls: List[str] = []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.error("Failed to parse sitemap XML: %s", exc)
        sys.exit(1)

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Sitemap index → fetch sub-sitemaps
    for sitemap_tag in root.findall(".//sm:sitemap/sm:loc", ns):
        sub_url = sitemap_tag.text.strip()
        logger.debug("Found sub-sitemap: %s", sub_url)
        sub_resp = fetch_with_retry(session, sub_url)
        if sub_resp is None:
            continue
        try:
            sub_root = ET.fromstring(sub_resp.content)
            for loc in sub_root.findall(".//sm:url/sm:loc", ns):
                u = loc.text.strip()
                if is_doc_url(u):
                    urls.append(u)
        except ET.ParseError as exc:
            logger.error("Failed to parse sub-sitemap %s: %s", sub_url, exc)

    # Flat sitemap
    for loc in root.findall(".//sm:url/sm:loc", ns):
        u = loc.text.strip()
        if is_doc_url(u) and u not in urls:
            urls.append(u)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    logger.info("Found %d documentation URLs in sitemap", len(unique))
    return unique


def filter_by_sections(urls: List[str], sections: List[str]) -> List[str]:
    """Keep only URLs whose path starts with one of the requested section prefixes."""
    from urllib.parse import urlparse
    result: List[str] = []
    for u in urls:
        path = urlparse(u).path.strip("/")
        if any(path.startswith(s) for s in sections):
            result.append(u)
    return result


# ---------------------------------------------------------------------------
# Page metadata extraction
# ---------------------------------------------------------------------------

def extract_page_metadata(html: str, url: str) -> Tuple[str, str]:
    """
    Return (page_title, section) for a page.

    Tries multiple selectors before falling back to sensible defaults.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    for sel in SELECTORS["page_title"]:
        tag_name = sel.split("#")[0].split(".")[0]
        attrs: Dict = {}
        if "#" in sel:
            attrs["id"] = sel.split("#")[1].split(".")[0]
        if "." in sel:
            attrs["class"] = sel.split(".")[-1]
        tag = soup.find(tag_name, **attrs) if attrs else soup.find(tag_name)
        if tag:
            title = tag.get_text(strip=True)
            break

    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True).split("|")[0].strip()
    if not title:
        title = url.rstrip("/").split("/")[-1].replace("-", " ").title()

    # Section from eyebrow or URL
    section = ""
    for sel in SELECTORS["section_eyebrow"]:
        tag_name, *rest = sel.split(".")
        cls = rest[0] if rest else None
        tag = soup.find(tag_name, class_=cls) if cls else soup.find(tag_name)
        if tag:
            section = tag.get_text(strip=True)
            break
    if not section:
        section = url_to_section(url)

    return title, section


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> Dict:
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_progress(data: Dict) -> None:
    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Core scrape loop
# ---------------------------------------------------------------------------

def scrape_pages(
    urls: List[str],
    output_file: str,
    already_scraped: Set[str],
    save_html: bool = False,
    debug: bool = False,
) -> Dict:
    """
    Iterate over URLs, scrape each page, write chunks to JSONL.

    Returns a stats dict.
    """
    session = _make_session()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if save_html:
        Path(RAW_HTML_DIR).mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(urls),
        "scraped": 0,
        "skipped": 0,
        "errors": 0,
        "chunks_written": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    pending = [u for u in urls if u not in already_scraped]
    logger.info("%d URLs to scrape (%d already done)", len(pending), len(already_scraped))

    buffer: List = []

    with tqdm(total=len(pending), unit="page", desc="Scraping") as pbar:
        for i, url in enumerate(pending):
            pbar.set_postfix_str(url.replace(BASE_URL, ""))
            logger.debug("Fetching %s", url)

            time.sleep(RATE_LIMIT_DELAY)
            resp = fetch_with_retry(session, url)

            if resp is None:
                stats["errors"] += 1
                pbar.update(1)
                continue

            html = resp.text

            # Optionally save raw HTML for debugging
            if save_html:
                slug = url.replace(BASE_URL, "").strip("/").replace("/", "_") or "index"
                html_path = os.path.join(RAW_HTML_DIR, f"{slug}.html")
                with open(html_path, "w", encoding="utf-8") as fh:
                    fh.write(html)

            try:
                page_title, section = extract_page_metadata(html, url)
                chunks = parse_page_to_chunks(html, url, page_title, section)

                if not chunks:
                    logger.debug("No chunks extracted from %s", url)
                    stats["skipped"] += 1
                    pbar.update(1)
                    continue

                buffer.extend(chunks)
                stats["scraped"] += 1
                stats["chunks_written"] += len(chunks)
                logger.debug("  → %d chunks from '%s'", len(chunks), page_title)

            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error processing %s:\n%s", url, traceback.format_exc())
                stats["errors"] += 1
                pbar.update(1)
                continue

            # Flush buffer every N pages
            if (i + 1) % SAVE_EVERY_N_PAGES == 0 and buffer:
                written = write_chunks_jsonl(buffer, output_file, mode="a")
                logger.info(
                    "Progress: %d/%d pages — %d chunks written so far",
                    stats["scraped"], len(pending), stats["chunks_written"],
                )
                buffer.clear()
                save_progress(stats)

            pbar.update(1)

    # Flush remaining buffer
    if buffer:
        write_chunks_jsonl(buffer, output_file, mode="a")

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_progress(stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Pipecat documentation into chunks.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_FILE,
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip URLs already present in the output file",
    )
    parser.add_argument(
        "--sections",
        default="",
        help="Comma-separated list of sections to scrape "
             f"(options: {', '.join(KNOWN_SECTIONS)}). Default: all.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        dest="save_html",
        help=f"Save raw HTML to {RAW_HTML_DIR}/ for debugging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Fetch sitemap and show stats; do not scrape pages",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    session = _make_session()
    all_urls = fetch_sitemap_urls(session)

    # Filter by section if requested
    if args.sections:
        requested = [s.strip() for s in args.sections.split(",") if s.strip()]
        invalid = [s for s in requested if s not in KNOWN_SECTIONS]
        if invalid:
            logger.warning("Unknown sections (will be ignored): %s", invalid)
        valid = [s for s in requested if s in KNOWN_SECTIONS]
        all_urls = filter_by_sections(all_urls, valid)
        logger.info("Filtered to %d URLs for sections: %s", len(all_urls), valid)

    # Show first 10 URLs in all modes
    logger.info("--- First 10 URLs ---")
    for u in all_urls[:10]:
        logger.info("  %s", u)
    logger.info("--- Total: %d URLs ---", len(all_urls))

    if args.dry_run:
        # Categorise by section and print summary
        from urllib.parse import urlparse
        from collections import Counter
        counts: Counter = Counter()
        for u in all_urls:
            path = urlparse(u).path.strip("/")
            matched = next((s for s in KNOWN_SECTIONS if path.startswith(s)), "other")
            counts[matched] += 1
        print("\nDry-run — pages per section:")
        for section, count in sorted(counts.items(), key=lambda x: -x[1]):
            label = SECTION_MAP.get(section, section)
            print(f"  {label:<20} {count:>4} pages")
        print(f"\n  {'TOTAL':<20} {len(all_urls):>4} pages")
        return

    # Resume: skip already-scraped URLs
    already_scraped: Set[str] = set()
    if args.resume or Path(args.output).exists():
        already_scraped = load_scraped_urls(args.output)
        if already_scraped:
            logger.info("Resuming — %d URLs already in output file", len(already_scraped))

    stats = scrape_pages(
        urls=all_urls,
        output_file=args.output,
        already_scraped=already_scraped,
        save_html=args.save_html,
        debug=args.debug,
    )

    print("\n=== Scrape complete ===")
    print(f"  Pages scraped:   {stats['scraped']}")
    print(f"  Pages skipped:   {stats['skipped']}")
    print(f"  Errors:          {stats['errors']}")
    print(f"  Chunks written:  {stats['chunks_written']}")
    print(f"  Output:          {args.output}")


if __name__ == "__main__":
    main()
