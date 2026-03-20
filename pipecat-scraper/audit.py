"""
Audit script: fetch sample pages from each section, compare raw HTML
elements against extracted chunks, and flag anything potentially missing.

Usage:
    python3 audit.py                      # audit default sample URLs
    python3 audit.py --url <url>          # audit a single specific URL
    python3 audit.py --save-html          # also save raw HTML for manual inspection
"""

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag

from config import HEADERS
from utils import parse_page_to_chunks, _find_content_div, _flatten_content

# ---------------------------------------------------------------------------
# Sample URLs — one or two per section (adjust as needed)
# ---------------------------------------------------------------------------
SAMPLE_URLS = [
    # Get Started
    "https://docs.pipecat.ai/getting-started/introduction",
    # Guides — has lots of Python code
    "https://docs.pipecat.ai/guides/learn/pipeline",
    # Server — API reference style
    "https://docs.pipecat.ai/server/introduction",
    # Client
    "https://docs.pipecat.ai/client/introduction",
    # CLI — has code + tables
    "https://docs.pipecat.ai/cli/cloud/agent",
    # Deployment
    "https://docs.pipecat.ai/deployment/overview",
    # Examples
    "https://docs.pipecat.ai/examples",
]

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# HTML inventory helpers
# ---------------------------------------------------------------------------

def inventory_html(html: str) -> Dict:
    """
    Count every interesting HTML element in the raw page (whole document)
    and inside the content div specifically.
    """
    soup = BeautifulSoup(html, "lxml")
    content_div = _find_content_div(soup)

    def count_tags(root: Optional[Tag]) -> Dict:
        if root is None:
            return {}
        counts: Counter = Counter()
        for tag in root.find_all(True):
            counts[tag.name] += 1
        return dict(counts.most_common())

    # Code-block specific checks
    pre_tags     = content_div.find_all("pre")       if content_div else []
    code_tags    = content_div.find_all("code")      if content_div else []
    figures      = content_div.find_all("figure")    if content_div else []
    tables       = content_div.find_all("table")     if content_div else []
    lists        = content_div.find_all(["ul","ol"]) if content_div else []
    h2s          = content_div.find_all("h2")        if content_div else []
    h3s          = content_div.find_all("h3")        if content_div else []

    # Detect orphaned text that looks like code (contains indented lines)
    orphaned_code_hints: List[str] = []
    if content_div:
        flat = _flatten_content(content_div)
        for el in flat:
            if not isinstance(el, Tag):
                continue
            if el.name in ("p", "div"):
                txt = el.get_text()
                lines = txt.splitlines()
                indented = sum(1 for l in lines if l.startswith("    ") or l.startswith("\t"))
                if indented >= 2 and el.name != "pre":
                    orphaned_code_hints.append(txt[:120].strip())

    return {
        "content_div_found":  content_div is not None,
        "content_div_tag":    f"<{content_div.name} class='{' '.join(content_div.get('class',[]))}'>"
                               if content_div else None,
        "h2_count":           len(h2s),
        "h3_count":           len(h3s),
        "pre_count":          len(pre_tags),
        "code_tag_count":     len(code_tags),
        "figure_count":       len(figures),
        "table_count":        len(tables),
        "list_count":         len(lists),
        "h2_texts":           [h.get_text(strip=True)[:60] for h in h2s],
        "orphaned_code_hints": orphaned_code_hints[:3],
    }


def inventory_chunks(chunks: List[Dict]) -> Dict:
    """Summarise the extracted chunks."""
    types: Counter = Counter(c["content_type"] for c in chunks)
    code_langs: Counter = Counter(
        c["code_language"] or "unknown"
        for c in chunks if c["content_type"] == "code"
    )
    h2_coverage = sorted({c["h2"] for c in chunks if c["h2"]})

    return {
        "total_chunks":  len(chunks),
        "by_type":       dict(types),
        "code_languages": dict(code_langs),
        "h2_coverage":   h2_coverage,
    }


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def analyse_gaps(html_inv: Dict, chunk_inv: Dict) -> List[str]:
    """Return a list of warning strings where extraction may be incomplete."""
    warnings: List[str] = []

    if not html_inv["content_div_found"]:
        warnings.append("CRITICAL: content div not found — zero chunks extracted")
        return warnings

    # Code blocks
    pre_count   = html_inv["pre_count"]
    code_chunks = chunk_inv["by_type"].get("code", 0)
    if pre_count > 0 and code_chunks == 0:
        warnings.append(
            f"MISSING CODE: {pre_count} <pre> tag(s) in HTML but 0 code chunks extracted"
        )
    elif pre_count > code_chunks * 2:
        warnings.append(
            f"POSSIBLE MISSING CODE: {pre_count} <pre> tags but only {code_chunks} code chunks"
        )

    # Tables
    tbl_count  = html_inv["table_count"]
    tbl_chunks = chunk_inv["by_type"].get("table", 0)
    if tbl_count > 0 and tbl_chunks == 0:
        warnings.append(
            f"MISSING TABLES: {tbl_count} <table> element(s) in HTML but 0 table chunks"
        )

    # Lists
    lst_count  = html_inv["list_count"]
    lst_chunks = chunk_inv["by_type"].get("list", 0)
    if lst_count > 0 and lst_chunks == 0:
        warnings.append(
            f"MISSING LISTS: {lst_count} list(s) in HTML but 0 list chunks"
        )

    # Headings vs chunk h2 coverage
    # Some h2s are structural card-group headers — their content flows into
    # nested sub-h2 chunks (cards with their own headings). We only flag a
    # coverage problem when fewer than 60% of HTML h2s are represented in chunks,
    # which indicates whole sections are genuinely missing.
    h2_in_html_count   = html_inv["h2_count"]
    h2_in_chunks_count = len(chunk_inv["h2_coverage"])
    if h2_in_html_count > 0:
        coverage_ratio = h2_in_chunks_count / h2_in_html_count
        if coverage_ratio < 0.60:
            h2_in_chunks_clean = {h.lstrip("\u200b").strip() for h in chunk_inv["h2_coverage"]}
            missed = [
                h.lstrip("\u200b").strip()
                for h in html_inv["h2_texts"]
                if h.lstrip("\u200b").strip() not in h2_in_chunks_clean
            ]
            warnings.append(
                f"LOW H2 COVERAGE ({h2_in_chunks_count}/{h2_in_html_count} = "
                f"{coverage_ratio:.0%}): possibly missing sections: {missed[:5]}"
            )

    # Orphaned code-like text
    if html_inv["orphaned_code_hints"]:
        warnings.append(
            f"POSSIBLE ORPHANED CODE in plain text ({len(html_inv['orphaned_code_hints'])} hint(s)): "
            + repr(html_inv["orphaned_code_hints"][0][:80])
        )

    if not warnings:
        warnings.append("OK — no obvious gaps detected")

    return warnings


# ---------------------------------------------------------------------------
# Sample chunk printer
# ---------------------------------------------------------------------------

def print_sample_chunks(chunks: List[Dict], n_per_type: int = 1) -> None:
    seen_types: set = set()
    for c in chunks:
        t = c["content_type"]
        if t in seen_types:
            continue
        seen_types.add(t)
        preview = c["text"][:300].replace("\n", "\\n")
        lang = f" [{c['code_language']}]" if c["code_language"] else ""
        print(f"  [{t}{lang}]  h2='{c['h2']}'  h3='{c['h3']}'")
        print(f"    {preview}")
        print()
        if len(seen_types) >= n_per_type * 3:
            break


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

def audit_url(url: str, session: requests.Session, save_html: bool = False) -> bool:
    """
    Audit one URL. Returns True if no critical issues found.
    """
    print(SEP)
    print(f"URL: {url}")
    print(SEP)

    resp = session.get(url, timeout=15)
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} — skipping")
        return False

    html = resp.text

    if save_html:
        slug = url.split("docs.pipecat.ai/")[-1].replace("/", "_") or "index"
        path = Path("data/raw_docs") / f"{slug}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        print(f"  Saved HTML → {path}")

    html_inv   = inventory_html(html)
    chunks     = parse_page_to_chunks(html, url, "Audit", "Audit")
    chunk_inv  = inventory_chunks(chunks)
    gaps       = analyse_gaps(html_inv, chunk_inv)

    # ---- HTML inventory ----
    print(f"\n[HTML inventory]")
    print(f"  Content div : {html_inv['content_div_tag']}")
    print(f"  h2 / h3     : {html_inv['h2_count']} / {html_inv['h3_count']}")
    print(f"  <pre> tags  : {html_inv['pre_count']}")
    print(f"  <code> tags : {html_inv['code_tag_count']}")
    print(f"  <figure>    : {html_inv['figure_count']}")
    print(f"  <table>     : {html_inv['table_count']}")
    print(f"  <ul>/<ol>   : {html_inv['list_count']}")
    if html_inv["h2_texts"]:
        print(f"  H2 sections : {html_inv['h2_texts']}")

    # ---- Chunk inventory ----
    print(f"\n[Chunk inventory]")
    print(f"  Total chunks : {chunk_inv['total_chunks']}")
    print(f"  By type      : {chunk_inv['by_type']}")
    if chunk_inv["code_languages"]:
        print(f"  Code langs   : {chunk_inv['code_languages']}")
    if chunk_inv["h2_coverage"]:
        print(f"  H2 covered   : {chunk_inv['h2_coverage']}")

    # ---- Gap analysis ----
    print(f"\n[Gap analysis]")
    ok = True
    for w in gaps:
        icon = "✓" if w.startswith("OK") else "⚠"
        print(f"  {icon} {w}")
        if not w.startswith("OK"):
            ok = False

    # ---- Sample chunks ----
    print(f"\n[Sample chunks — one per type]")
    print_sample_chunks(chunks, n_per_type=1)

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Pipecat scraper extraction quality")
    parser.add_argument("--url",       help="Audit a single URL instead of the default sample set")
    parser.add_argument("--save-html", action="store_true", dest="save_html",
                        help="Save raw HTML files to data/raw_docs/")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    urls = [args.url] if args.url else SAMPLE_URLS

    results: Dict[str, bool] = {}
    for url in urls:
        ok = audit_url(url, session, save_html=args.save_html)
        results[url] = ok

    print(SEP)
    print("AUDIT SUMMARY")
    print(SEP)
    passed = sum(results.values())
    print(f"  {passed}/{len(results)} pages passed with no warnings\n")
    for url, ok in results.items():
        icon = "✓" if ok else "⚠"
        short = url.replace("https://docs.pipecat.ai/", "")
        print(f"  {icon}  {short}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
