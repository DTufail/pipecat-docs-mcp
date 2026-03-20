"""
Configuration and constants for the Pipecat documentation scraper.
"""
from typing import Dict, List

# Base URL and sitemap
BASE_URL = "https://docs.pipecat.ai"
SITEMAP_URL = "https://docs.pipecat.ai/sitemap.xml"

# HTTP headers to mimic a browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Rate limiting
RATE_LIMIT_DELAY = 0.5       # seconds between requests
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0         # exponential backoff multiplier
RETRY_STATUS_CODES = {429, 503, 502, 504}

# Progress saving
SAVE_EVERY_N_PAGES = 10

# Section mappings: URL path prefix → human-readable section name
SECTION_MAP: Dict[str, str] = {
    "getting-started": "Get Started",
    "guides":          "Guides",
    "server":          "Server",
    "client":          "Client",
    "cli":             "CLI",
    "deployment":      "Deployment",
    "examples":        "Examples",
    "api-reference":   "API Reference",
    "open-source":     "Open Source",
    "changelog":       "Changelog",
    "concepts":        "Concepts",
}

# All known section prefixes (used for URL categorisation)
KNOWN_SECTIONS: List[str] = list(SECTION_MAP.keys())

# CSS / tag selectors (Mintlify-specific; parameterise here for easy porting)
SELECTORS = {
    "page_title":  ["h1", "h1#page-title"],
    "section_eyebrow": ["div.eyebrow", "span.eyebrow"],
    "main_content": [
        "div#content.mdx-content",
        "div.mdx-content",
        "article",
        "main",
    ],
    "description_meta": "meta[name='description']",
}

# File paths (relative to project root)
OUTPUT_FILE     = "data/chunks.jsonl"
RAW_HTML_DIR    = "data/raw_docs"
ERROR_LOG_FILE  = "errors.log"
PROGRESS_FILE   = "data/.progress.json"
