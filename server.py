"""
Pipecat Documentation MCP Server

Exposes 4 tools over 317 pages of Pipecat voice AI framework documentation.

Run (Claude Desktop / stdio):
    python server.py

Run (HTTP for Cursor / remote):
    fastmcp run server.py:mcp --transport http --port 8000
"""

import os
# Must be before torch / tokenizers are imported
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

from fastmcp import FastMCP
from retrieval import search

mcp = FastMCP(
    name="pipecat-docs",
    instructions=(
        "Searches 317 pages of Pipecat voice AI framework documentation. "
        "Use search_pipecat_docs for general questions, get_example_code "
        "when you need runnable Python, explain_concept for architecture "
        "questions, and compare_services when choosing between providers."
    ),
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _breadcrumb(r: dict) -> str:
    return " > ".join(p for p in [r.get("section"), r.get("h2"), r.get("h3")] if p)


# ── Tool 1: General search ────────────────────────────────────────────────────

@mcp.tool
def search_pipecat_docs(query: str, max_results: int = 5, mode: str = "hybrid") -> str:
    """Search Pipecat documentation by natural language query.
    Use for configuration questions, error messages, and general how-to queries."""
    max_results = min(max_results, 10)
    preview_len = 200 if max_results >= 8 else 400

    try:
        results = search(query, mode=mode, top_k=max_results)
    except Exception as e:
        return f"Retrieval error: {e}. Ensure indexes are built (run indexer.py)."

    if not results:
        return f"No documentation found for '{query}'. Try rephrasing or use a more specific term."

    parts = []
    total_chars = 0
    for r in results:
        crumb = _breadcrumb(r)
        text  = r["text"]
        snip  = text[:preview_len] + ("…" if len(text) > preview_len else "")
        block = f"### {crumb}\n[{r['content_type']}] {snip}\nSource: {r['source_url']}"
        total_chars += len(block)
        if total_chars > 2000:
            break
        parts.append(block)

    return "\n\n---\n\n".join(parts)


# ── Tool 2: Code examples ─────────────────────────────────────────────────────

@mcp.tool
def get_example_code(use_case: str, max_examples: int = 2) -> str:
    """Retrieve complete, runnable Pipecat pipeline code examples.
    Use when the user needs working Python code, not documentation prose."""
    try:
        results = search(f"example {use_case}", top_k=max_examples * 5, mode="code")
    except Exception as e:
        return f"Retrieval error: {e}. Ensure indexes are built (run indexer.py)."

    code_results = [r for r in results if r.get("content_type") == "code"]
    if not code_results:
        return f"No code examples found for '{use_case}'. Try search_pipecat_docs instead."

    # Prefer longer (more complete) examples
    code_results.sort(key=lambda r: len(r["text"]), reverse=True)
    code_results = code_results[:max_examples]

    parts = []
    total_chars = 0
    for n, r in enumerate(code_results, 1):
        lang  = r.get("code_language") or "python"
        title = r.get("page_title") or ""
        h2    = r.get("h2") or _breadcrumb(r)
        text  = r["text"]
        truncated = len(text) > 1500
        snip  = text[:1500] if truncated else text
        suffix = f"\n# [truncated — full example at {r['source_url']}]" if truncated else ""
        block = (
            f"## Example {n}: {title} — {h2}\n"
            f"Source: {r['source_url']}\n\n"
            f"```{lang}\n{snip}{suffix}\n```"
        )
        total_chars += len(block)
        if total_chars > 3000:
            break
        parts.append(block)

    return "\n\n".join(parts)


# ── Tool 3: Concept explanation ───────────────────────────────────────────────

@mcp.tool
def explain_concept(concept: str) -> str:
    """Explain a Pipecat architecture concept (frames, processors, pipelines, transports, VAD).
    Use for 'what is X' and 'how does X work' questions."""
    try:
        results = search(concept, top_k=6, mode="semantic")
    except Exception as e:
        return f"Retrieval error: {e}. Ensure indexes are built (run indexer.py)."

    if not results:
        return f"No documentation found for concept '{concept}'. Try a related term."

    # Drop garbage chunks (MDX fragments, template syntax, etc.)
    results = [r for r in results if len(r["text"].strip()) >= 30]

    # Prefer text chunks; boost foundational sections; deduplicate by h2
    _PRIORITY = {"Learning Pipecat", "Guides", "Get Started"}

    def _rank_key(r):
        section_penalty = 0 if r.get("section") in _PRIORITY else 1
        type_penalty    = 0 if r.get("content_type") == "text" else 1
        return (section_penalty, type_penalty)

    seen_h2: set[str] = set()
    filtered = []
    for r in sorted(results, key=_rank_key):
        h2 = r.get("h2") or ""
        if h2 not in seen_h2:
            seen_h2.add(h2)
            filtered.append(r)

    parts = [f"## {concept} — Explanation from Pipecat Docs\n"]
    total_chars = len(parts[0])
    for r in filtered:
        h2      = r.get("h2") or r.get("section") or ""
        section = r.get("section") or ""
        text    = r["text"]
        snip    = text[:300] + ("…" if len(text) > 300 else "")
        block   = f"**{h2}** ({section})\n{snip}\nSource: {r['source_url']}"
        total_chars += len(block)
        if total_chars > 1500:
            break
        parts.append(block)

    return "\n\n---\n\n".join(parts)


# ── Tool 4: Service comparison ────────────────────────────────────────────────

@mcp.tool
def compare_services(service_a: str, service_b: str, aspect: str = "") -> str:
    """Compare two Pipecat-compatible services side by side (e.g. STT providers, TTS providers).
    Use when choosing between Deepgram vs AssemblyAI, ElevenLabs vs Cartesia, etc."""
    query_a = f"{service_a} {aspect}".strip()
    query_b = f"{service_b} {aspect}".strip()

    try:
        results_a = search(query_a, top_k=3, mode="hybrid")
        results_b = search(query_b, top_k=3, mode="hybrid")
    except Exception as e:
        return f"Retrieval error: {e}. Ensure indexes are built (run indexer.py)."

    aspect_label = f" — {aspect}" if aspect else ""
    lines = [f"## {service_a} vs {service_b}{aspect_label}\n"]

    for label, results in [(service_a, results_a), (service_b, results_b)]:
        lines.append(f"### {label}")
        if not results:
            lines.append(f"- No documentation found for {label}.")
        else:
            for r in results:
                snip = r["text"][:200].replace("\n", " ")
                lines.append(f"- {snip}\n  Source: {r['source_url']}")
        lines.append("")

    output = "\n".join(lines)
    return output[:1500]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
