"""
Fetch all GitHub Issues from pipecat-ai/pipecat and convert them to
retrieval chunks compatible with the existing docs index schema.

Output: data/github_issues.jsonl  (one JSON object per line)

Usage:
    python github_indexer.py
    python github_indexer.py --token ghp_xxxx --max 50
    python github_indexer.py --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO       = "pipecat-ai/pipecat"
API_BASE   = "https://api.github.com"
MAX_COMMENTS = 5       # comments fetched per issue
COMMENT_SLEEP = 0.1   # seconds between comment requests
RETRY_MAX   = 3


# ── HTTP helpers ─────────────────────────────────────────────────────────────


def _make_session(token: str | None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/vnd.github+json"})
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _get_with_retry(session: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    """GET with automatic rate-limit handling and exponential-backoff retry."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt == RETRY_MAX:
                raise
            wait = 2 ** attempt
            log.warning("Network error (%s), retry %d/%d in %ds", exc, attempt, RETRY_MAX, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code == 403:
            reset_ts = resp.headers.get("X-RateLimit-Reset")
            if reset_ts:
                wait = max(int(reset_ts) - int(time.time()) + 1, 1)
                log.warning("Rate limited. Sleeping %ds until reset …", wait)
                time.sleep(wait)
                continue
            # Non-rate-limit 403 (bad token etc.)
            log.error("403 Forbidden: %s", resp.text[:200])
            resp.raise_for_status()

        if resp.status_code == 404:
            return resp   # caller handles 404 by skipping

        if attempt == RETRY_MAX:
            resp.raise_for_status()

        wait = 2 ** attempt
        log.warning("HTTP %d, retry %d/%d in %ds", resp.status_code, attempt, RETRY_MAX, wait)
        time.sleep(wait)

    raise RuntimeError(f"Failed to GET {url} after {RETRY_MAX} retries")


# ── Fetch issues ─────────────────────────────────────────────────────────────


def fetch_all_issues(session: requests.Session, max_issues: int | None) -> list[dict]:
    """Paginate through all issues (open + closed), skip pull requests."""
    issues: list[dict] = []
    page = 1

    while True:
        log.info("Fetching page %d … (%d issues so far)", page, len(issues))
        resp = _get_with_retry(
            session,
            f"{API_BASE}/repos/{REPO}/issues",
            params={"state": "all", "per_page": 100, "page": page},
        )

        data = resp.json()
        if not data:
            break  # empty page → done

        for item in data:
            if "pull_request" in item:
                continue   # GitHub issues API includes PRs — skip them
            issues.append(item)
            if max_issues and len(issues) >= max_issues:
                log.info("Reached --max %d, stopping.", max_issues)
                return issues

        page += 1

    return issues


def fetch_comments(session: requests.Session, issue: dict) -> list[dict]:
    """Fetch up to MAX_COMMENTS comments for a single issue."""
    if not issue.get("comments", 0):
        return []

    time.sleep(COMMENT_SLEEP)
    resp = _get_with_retry(
        session,
        issue["comments_url"],
        params={"per_page": MAX_COMMENTS},
    )
    if resp.status_code == 404:
        return []
    return resp.json()[:MAX_COMMENTS]


# ── Build chunks ─────────────────────────────────────────────────────────────


def _label_names(issue: dict) -> str:
    return ", ".join(lb["name"] for lb in issue.get("labels", []))


def issue_to_chunks(issue: dict, comments: list[dict]) -> list[dict]:
    """Convert one GitHub issue + its comments into 1–2 retrieval chunks."""
    chunks: list[dict] = []

    number   = issue["number"]
    title    = issue.get("title") or ""
    body     = (issue.get("body") or "").strip()
    url      = issue["html_url"]
    labels   = _label_names(issue)

    # Skip issues with no meaningful body
    if len(body) < 30:
        body = ""

    # Chunk 1 — issue body
    body_text = f"Title: {title}\n\n{body[:1000]}" if body else f"Title: {title}"
    chunks.append({
        "chunk_id":     f"gh_issue_{number}_body",
        "source_url":   url,
        "page_title":   f"Issue #{number}: {title}",
        "section":      "GitHub Issues",
        "h2":           labels or "Unlabeled",
        "h3":           None,
        "content_type": "issue",
        "text":         body_text,
        "code_language": None,
        "chunk_index":  0,
    })

    # Chunk 2 — top comments (only if substantive)
    substantive = [c for c in comments if len((c.get("body") or "").strip()) > 50]
    if substantive:
        parts = []
        for c in substantive[:3]:
            author = c.get("user", {}).get("login", "unknown")
            text   = (c.get("body") or "").strip()[:500]
            parts.append(f"@{author}: {text}")
        chunks.append({
            "chunk_id":     f"gh_issue_{number}_comments",
            "source_url":   url,
            "page_title":   f"Issue #{number}: {title} — Comments",
            "section":      "GitHub Issues",
            "h2":           "Resolution",
            "h3":           None,
            "content_type": "issue_comment",
            "text":         "\n---\n".join(parts),
            "code_language": None,
            "chunk_index":  1,
        })

    return chunks


# ── Resume support ────────────────────────────────────────────────────────────


def load_existing_ids(output_path: Path) -> set[str]:
    """Return chunk_ids already written (for --resume)."""
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Index GitHub Issues from pipecat-ai/pipecat")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--output",
        default="data/github_issues.jsonl",
        help="Output path for issue chunks (default: data/github_issues.jsonl)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Fetch at most N issues (useful for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip issues whose chunks are already in the output file",
    )
    args = parser.parse_args()

    if not args.token:
        log.warning(
            "No GitHub token found. Unauthenticated requests are limited to "
            "60/hour. Pass --token or set GITHUB_TOKEN."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session(args.token)

    # --- Fetch all issues ---
    issues = fetch_all_issues(session, args.max)
    log.info("Fetched %d issues total", len(issues))

    # --- Resume: find already-indexed issue numbers ---
    skip_numbers: set[int] = set()
    if args.resume:
        existing_ids = load_existing_ids(output_path)
        for issue in issues:
            body_id = f"gh_issue_{issue['number']}_body"
            if body_id in existing_ids:
                skip_numbers.add(issue["number"])
        log.info("Resuming: %d issues already indexed, skipping", len(skip_numbers))

    # --- Process and write ---
    mode = "a" if args.resume else "w"
    total_chunks = 0
    total_issues = 0

    with open(output_path, mode) as out:
        for i, issue in enumerate(issues):
            if issue["number"] in skip_numbers:
                continue

            comments = fetch_comments(session, issue)
            chunks   = issue_to_chunks(issue, comments)

            for chunk in chunks:
                out.write(json.dumps(chunk) + "\n")

            total_chunks += len(chunks)
            total_issues += 1

            if (i + 1) % 50 == 0:
                log.info("Processed %d / %d issues (%d chunks written)", i + 1, len(issues), total_chunks)

    log.info(
        "Done. %d issues → %d chunks written to %s",
        total_issues, total_chunks, output_path,
    )
    log.info("Run 'python indexer.py' to rebuild the search indexes with GitHub Issues included.")


if __name__ == "__main__":
    main()
