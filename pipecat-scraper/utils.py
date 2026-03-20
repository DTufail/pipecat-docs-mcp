"""
Helper functions: content parsing and semantic chunking.
"""
import re
import json
import logging
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from config import BASE_URL, SECTION_MAP, KNOWN_SECTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def url_to_section(url: str) -> str:
    """Return the human-readable section name for a documentation URL."""
    path = urlparse(url).path.strip("/")
    for prefix, name in SECTION_MAP.items():
        if path.startswith(prefix):
            return name
    return "Documentation"


def url_to_chunk_prefix(url: str) -> str:
    """Return a slug prefix derived from the URL path (used in chunk_id)."""
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
    return slug or "doc"


def is_doc_url(url: str) -> bool:
    """Return True if the URL looks like a documentation page (not an asset)."""
    parsed = urlparse(url)
    if parsed.netloc and BASE_URL.split("//")[-1] not in parsed.netloc:
        return False
    path = parsed.path.lower()
    # Skip obvious non-page resources
    skip_extensions = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                       ".css", ".js", ".woff", ".woff2", ".ttf", ".pdf",
                       ".zip", ".tar", ".gz")
    if any(path.endswith(ext) for ext in skip_extensions):
        return False
    return True


# ---------------------------------------------------------------------------
# HTML → plain text helpers
# ---------------------------------------------------------------------------

def table_to_markdown(table: Tag) -> str:
    """Convert a <table> element to a compact Markdown table string."""
    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(separator=" ", strip=True)
                 for cell in tr.find_all(["th", "td"])]
        rows.append(cells)
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    # Pad rows
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    lines = [" | ".join(rows[0])]
    lines.append(" | ".join(["---"] * col_count))
    for row in rows[1:]:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def list_to_text(lst: Tag, indent: int = 0) -> str:
    """Recursively convert <ul>/<ol> to plain-text with indentation."""
    lines: List[str] = []
    ordered = lst.name == "ol"
    for i, li in enumerate(lst.find_all("li", recursive=False), start=1):
        # Extract direct text (exclude nested lists)
        text_parts: List[str] = []
        for child in li.children:
            if isinstance(child, NavigableString):
                t = str(child).strip()
                if t:
                    text_parts.append(t)
            elif child.name not in ("ul", "ol"):
                t = child.get_text(separator=" ", strip=True)
                if t:
                    text_parts.append(t)
        bullet = f"{i}." if ordered else "-"
        prefix = "  " * indent
        lines.append(f"{prefix}{bullet} {' '.join(text_parts)}")
        # Recurse into nested lists
        for nested in li.find_all(["ul", "ol"], recursive=False):
            lines.append(list_to_text(nested, indent + 1))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def parse_page_to_chunks(
    html: str,
    url: str,
    page_title: str,
    section: str,
) -> List[Dict[str, Any]]:
    """
    Parse page HTML into a list of semantic chunk dicts.

    Chunking strategy:
    - One chunk per h2 section (text gathered until next h2)
    - Within an h2, further split at h3 boundaries
    - Code blocks always become their own chunk
    - Tables and lists become their own chunk of the appropriate type
    """
    soup = BeautifulSoup(html, "lxml")

    # Locate main content container
    content_div = _find_content_div(soup)
    if content_div is None:
        logger.warning("No content div found for %s", url)
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    current_h2: Optional[str] = None
    current_h3: Optional[str] = None
    text_buffer: List[str] = []

    def flush_text() -> None:
        nonlocal chunk_index
        combined = "\n\n".join(text_buffer).strip()
        if combined:
            chunks.append(_make_chunk(
                url=url, page_title=page_title, section=section,
                h2=current_h2, h3=current_h3,
                content_type="text", text=combined,
                code_language=None, index=chunk_index,
            ))
            chunk_index += 1
        text_buffer.clear()

    # Flatten the content tree into a linear stream of logical elements.
    # This handles code blocks that are nested inside wrapper divs
    # (Mintlify wraps <pre> in a div with a Copy button).
    elements = _flatten_content(content_div)

    for element in elements:
        if not isinstance(element, Tag):
            t = str(element).strip()
            if t:
                text_buffer.append(t)
            continue

        tag = element.name

        # ---- Heading boundaries ----
        if tag == "h2":
            flush_text()
            current_h2 = _clean_heading(element)
            current_h3 = None
            continue

        if tag == "h3":
            flush_text()
            current_h3 = _clean_heading(element)
            continue

        if tag in ("h4", "h5", "h6"):
            text_buffer.append(f"**{element.get_text(strip=True)}**")
            continue

        # ---- Code blocks ----
        if tag == "pre":
            flush_text()
            code_tag = element.find("code")
            code_text = code_tag.get_text() if code_tag else element.get_text()
            lang = _detect_code_language(code_tag or element)
            if code_text.strip():
                chunks.append(_make_chunk(
                    url=url, page_title=page_title, section=section,
                    h2=current_h2, h3=current_h3,
                    content_type="code", text=code_text.strip(),
                    code_language=lang, index=chunk_index,
                ))
                chunk_index += 1
            continue

        # Inline code inside a paragraph
        if tag == "code" and element.parent and element.parent.name != "pre":
            text_buffer.append(f"`{element.get_text(strip=True)}`")
            continue

        # ---- Tables ----
        if tag == "table":
            flush_text()
            md = table_to_markdown(element)
            if md:
                chunks.append(_make_chunk(
                    url=url, page_title=page_title, section=section,
                    h2=current_h2, h3=current_h3,
                    content_type="table", text=md,
                    code_language=None, index=chunk_index,
                ))
                chunk_index += 1
            continue

        # ---- Lists ----
        if tag in ("ul", "ol"):
            flush_text()
            lst_text = list_to_text(element)
            if lst_text.strip():
                chunks.append(_make_chunk(
                    url=url, page_title=page_title, section=section,
                    h2=current_h2, h3=current_h3,
                    content_type="list", text=lst_text,
                    code_language=None, index=chunk_index,
                ))
                chunk_index += 1
            continue

        # ---- Mintlify special components (Cards, Callouts, Steps, etc.) ----
        if _is_mintlify_component(element):
            flush_text()
            component_text = _extract_component_text(element)
            if component_text.strip():
                chunks.append(_make_chunk(
                    url=url, page_title=page_title, section=section,
                    h2=current_h2, h3=current_h3,
                    content_type="text", text=component_text,
                    code_language=None, index=chunk_index,
                ))
                chunk_index += 1
            continue

        # ---- Default: paragraphs and plain divs ----
        text = _clean_text(element)
        if text:
            text_buffer.append(text)

    flush_text()
    return chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _flatten_content(content_div: Tag) -> List:
    """
    Walk the content div and yield a flat stream of logical elements.

    Key behaviour:
    - Headings (h1-h6), pre, table, ul/ol → yielded as-is
    - Divs / figures that wrap a <pre> → yield the <pre> directly
      (discards copy-button siblings so "Copy" text doesn't leak into output)
    - Divs that wrap headings/lists → recurse into them
    - Everything else → yield as-is for default text extraction
    """
    result: List = []

    BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6",
                  "p", "pre", "table", "ul", "ol", "figure", "blockquote"}

    def _walk(node: Tag) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                result.append(child)
                continue

            tag = child.name

            # Always surface headings, pre, table, lists directly
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6",
                       "pre", "table", "ul", "ol"):
                result.append(child)
                continue

            # <figure data-rehype-pretty-code-figure> → surface the inner <pre>
            if tag == "figure" and child.has_attr("data-rehype-pretty-code-figure"):
                pre = child.find("pre")
                if pre:
                    # Copy language from figcaption or data attr if present
                    fig_cap = child.find("figcaption")
                    if fig_cap:
                        lang_hint = fig_cap.get_text(strip=True).lower()
                        # Attach as data attr so _detect_code_language can find it
                        pre["data-language"] = lang_hint
                    result.append(pre)
                continue

            # Generic div/section/article: check if it wraps a <pre>
            if tag in ("div", "section", "article"):
                # 1. Code block wrapper div (contains a <pre>) → surface the <pre>
                #    Do this first — even if the div has component classes, we want
                #    the code, not the wrapper.
                pre = child.find("pre")
                if pre:
                    result.append(pre)
                    continue
                # 2. Mintlify leaf component (card, callout, etc.) with no inner
                #    headings → yield as a unit so it creates a text chunk under the
                #    current h2 rather than disappearing into the recursion.
                has_inner_heading = bool(child.find(["h1","h2","h3","h4","h5","h6"]))
                if _is_mintlify_component(child) and not has_inner_heading:
                    result.append(child)
                    continue
                # 3. Structural wrapper → recurse to surface headings/lists inside
                has_block = any(
                    isinstance(c, Tag) and c.name in BLOCK_TAGS
                    for c in child.descendants
                )
                if has_block:
                    _walk(child)
                else:
                    result.append(child)
                continue

            # Paragraphs and everything else
            result.append(child)

    _walk(content_div)
    return result


def _clean_heading(tag: Tag) -> str:
    """Return heading text, stripping anchor/permalink child elements."""
    # Mintlify injects a zero-width-space + anchor link inside headings
    for a in tag.find_all("a", class_=lambda c: c and "anchor" in " ".join(c)):
        a.decompose()
    text = tag.get_text(strip=True)
    # Strip leading zero-width spaces Mintlify adds
    return text.lstrip("\u200b").strip()


def _clean_text(tag: Tag) -> str:
    """
    Extract readable text from a generic element, removing UI noise like
    copy-button labels that Mintlify injects next to code blocks.
    """
    # Remove button elements (Copy, etc.)
    for btn in tag.find_all("button"):
        btn.decompose()
    # Remove aria-hidden spans (decorative icons)
    for span in tag.find_all("span", attrs={"aria-hidden": True}):
        span.decompose()
    return tag.get_text(separator=" ", strip=True)


def _find_content_div(soup: BeautifulSoup) -> Optional[Tag]:
    """Locate the main content container using a list of candidate selectors."""
    candidates = [
        soup.find("div", id="content", class_=lambda c: c and "mdx-content" in c),
        soup.find("div", class_="mdx-content"),
        soup.find("div", id="content"),
        soup.find("article"),
        soup.find("main"),
    ]
    for c in candidates:
        if c:
            return c
    return None


def _detect_code_language(code_tag: Tag) -> Optional[str]:
    """
    Infer code language from multiple possible attributes/classes.

    Mintlify/Shiki renders: <pre language="python"><code language="python">
    Standard rehype:        <code class="language-python">
    Our helper sets:        <pre data-language="python">
    """
    # Check the tag itself and its parent <pre> for any of the known patterns
    candidates = [code_tag]
    if code_tag.parent and code_tag.parent.name == "pre":
        candidates.append(code_tag.parent)

    for tag in candidates:
        # Plain `language` attribute (Mintlify/Shiki)
        lang = tag.get("language")
        if lang:
            return str(lang).strip()
        # data-language attribute (our helper + some Mintlify versions)
        lang = tag.get("data-language")
        if lang:
            return str(lang).strip()
        # class="language-*" (standard rehype-highlight / prism)
        for cls in tag.get("class", []):
            if cls.startswith("language-"):
                return cls[len("language-"):]

    return None


def _is_mintlify_component(tag: Tag) -> bool:
    """Heuristic: Mintlify components often have custom class names."""
    cls_str = " ".join(tag.get("class", []))
    mintlify_patterns = [
        "card", "callout", "note", "warning", "tip", "info",
        "accordion", "steps", "tabs", "frame", "badge",
    ]
    return any(p in cls_str.lower() for p in mintlify_patterns)


def _extract_component_text(tag: Tag) -> str:
    """Extract readable text from Mintlify component blocks."""
    parts: List[str] = []
    title_tag = tag.find(["h3", "h4", "strong", "b"])
    if title_tag:
        parts.append(f"**{title_tag.get_text(strip=True)}**")
    # Get remaining text excluding the title
    body = tag.get_text(separator="\n", strip=True)
    if parts and parts[0].strip("*") in body:
        body = body.replace(parts[0].strip("*"), "", 1).strip()
    if body:
        parts.append(body)
    return "\n".join(parts)


def _make_chunk(
    url: str,
    page_title: str,
    section: str,
    h2: Optional[str],
    h3: Optional[str],
    content_type: str,
    text: str,
    code_language: Optional[str],
    index: int,
) -> Dict[str, Any]:
    """Construct a chunk dict and assign a deterministic chunk_id."""
    prefix = url_to_chunk_prefix(url)
    chunk_id = f"{prefix}-{index:03d}"
    return {
        "chunk_id":      chunk_id,
        "source_url":    url,
        "page_title":    page_title,
        "section":       section,
        "h2":            h2,
        "h3":            h3,
        "content_type":  content_type,
        "text":          text,
        "code_language": code_language,
        "chunk_index":   index,
    }


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_chunks_jsonl(chunks: List[Dict[str, Any]], filepath: str, mode: str = "a") -> None:
    """Append or write chunks to a JSONL file, validating each entry."""
    written = 0
    with open(filepath, mode, encoding="utf-8") as fh:
        for chunk in chunks:
            try:
                line = json.dumps(chunk, ensure_ascii=False)
                fh.write(line + "\n")
                written += 1
            except (TypeError, ValueError) as exc:
                logger.error("Invalid chunk skipped (%s): %s", chunk.get("chunk_id"), exc)
    return written


def load_scraped_urls(filepath: str) -> set:
    """Return the set of source_url values already present in the JSONL file."""
    seen: set = set()
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    seen.add(obj.get("source_url", ""))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return seen
