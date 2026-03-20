"""
Test suite for Pipecat documentation hybrid retrieval.

Runs a set of reference queries across all search modes and reports:
  - Top 3 results per query
  - Latency per query
  - MRR (Mean Reciprocal Rank) against expected sections / content types
  - Summary table

Usage:
    python test_retrieval.py
    python test_retrieval.py --benchmark --output results.csv
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import argparse
import csv
import time
from dataclasses import dataclass, field

from retrieval import search, precompute_query_vectors

# ── Test fixtures ─────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    query:            str
    expected_type:    str          # 'text', 'code', 'list'
    expected_section: str          # substring match against section/h2/h3
    mode:             str = "hybrid"
    description:      str = ""


TEST_CASES: list[TestCase] = [
    # Conceptual queries
    TestCase("What is a frame in Pipecat?",          "text", "Learning Pipecat",   "hybrid",   "core concept"),
    TestCase("Explain how pipelines work",            "text", "Pipeline",           "semantic", "architecture"),

    # Code queries
    TestCase("Show me a Deepgram STT example",        "code", "Speech-to-Text",     "code",     "STT code"),
    TestCase("How do I create a pipeline with OpenAI","code", "LLM",                "code",     "LLM pipeline"),

    # Configuration queries
    TestCase("How to configure Deepgram Nova-3 model","text", "Speech-to-Text",     "hybrid",   "STT config"),
    TestCase("DeepgramSTTService parameters",          "text", "Speech-to-Text",     "keyword",  "API params"),

    # Integration / transport queries
    TestCase("Daily transport setup",                  "text", "Transport",          "hybrid",   "transport setup"),
    TestCase("WebSocket transport example",            "code", "Transport",          "code",     "WS code"),

    # Error / troubleshooting queries
    TestCase("AttributeError DeepgramSTTService",     "text", "Guides",             "hybrid",   "error lookup"),
    TestCase("How to fix audio latency issues",        "text", "Deployment",         "hybrid",   "latency fix"),

    # Feature-specific
    TestCase("VAD voice activity detection setup",     "text", "Audio Processing",   "hybrid",   "VAD config"),
    TestCase("Function calling with OpenAI LLM",       "code", "LLM",                "code",     "function calling"),
    TestCase("TTS text to speech ElevenLabs",          "text", "Text-to-Speech",     "hybrid",   "TTS service"),
    TestCase("Pipecat MCP tool use",                   "text", "MCP",                "hybrid",   "MCP integration"),
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query:          str
    mode:           str
    top_results:    list[dict]
    latency_ms:     float
    mrr:            float
    recall_at_5:    bool
    found_type:     bool
    expected_type:  str
    expected_sect:  str


def _matches_section(results: list[dict], expected: str) -> tuple[int, bool]:
    """
    Return (rank_of_first_match, recall_at_5).
    Match is true if expected string appears in section, h2, h3 or source_url.
    """
    exp_low = expected.lower()
    for rank, r in enumerate(results, start=1):
        haystack = " ".join([
            r.get("section") or "",
            r.get("h2") or "",
            r.get("h3") or "",
            r.get("source_url") or "",
            r.get("page_title") or "",
        ]).lower()
        if exp_low in haystack:
            return rank, rank <= 5
    return 0, False


def _first_content_type(results: list[dict], expected_type: str) -> bool:
    """True if the first result matches the expected content type."""
    return bool(results) and results[0].get("content_type") == expected_type


def evaluate(tc: TestCase, top_k: int = 5, q_vec=None) -> QueryResult:
    t0 = time.perf_counter()
    results = search(tc.query, mode=tc.mode, top_k=top_k, q_vec=q_vec)
    latency_ms = (time.perf_counter() - t0) * 1000

    rank, recall = _matches_section(results, tc.expected_section)
    mrr = 1.0 / rank if rank > 0 else 0.0
    found_type = _first_content_type(results, tc.expected_type)

    return QueryResult(
        query=tc.query,
        mode=tc.mode,
        top_results=results,
        latency_ms=latency_ms,
        mrr=mrr,
        recall_at_5=recall,
        found_type=found_type,
        expected_type=tc.expected_type,
        expected_sect=tc.expected_section,
    )


# ── Printing ─────────────────────────────────────────────────────────────────

def print_result_block(tc: TestCase, qr: QueryResult) -> None:
    print(f"\n{'='*72}")
    print(f"Query   : {tc.query}")
    print(f"Mode    : {tc.mode}  |  Expected: section≈{tc.expected_section!r} type={tc.expected_type}")
    print(f"Latency : {qr.latency_ms:.0f}ms  |  MRR={qr.mrr:.3f}  |  Recall@5={'✓' if qr.recall_at_5 else '✗'}")
    print()
    for i, r in enumerate(qr.top_results[:3], 1):
        breadcrumb = " > ".join(p for p in [r.get("section"), r.get("h2"), r.get("h3")] if p)
        lang = f" [{r['code_language']}]" if r.get("code_language") else ""
        print(f"  {i}. [{r['score']:.4f}] {breadcrumb}{lang}  ({r['content_type']})")
        print(f"     {r['source_url']}")
        preview = r["text"][:200].replace("\n", " ")
        print(f"     {preview}{'…' if len(r['text'])>200 else ''}")


def print_summary_table(query_results: list[QueryResult]) -> None:
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    header = f"{'Query':<45} {'Mode':<8} {'MRR':>5} {'R@5':>4} {'ms':>6}"
    print(header)
    print("-" * 72)
    mrr_total = 0.0
    recall_total = 0
    latency_total = 0.0
    for qr in query_results:
        short_q = (qr.query[:42] + "…") if len(qr.query) > 42 else qr.query
        r5 = "✓" if qr.recall_at_5 else "✗"
        print(f"{short_q:<45} {qr.mode:<8} {qr.mrr:>5.3f} {r5:>4} {qr.latency_ms:>5.0f}ms")
        mrr_total    += qr.mrr
        recall_total += qr.recall_at_5
        latency_total += qr.latency_ms
    n = len(query_results)
    print("-" * 72)
    print(f"{'AVERAGE':<45} {'':8} {mrr_total/n:>5.3f} {recall_total/n*100:>3.0f}% {latency_total/n:>5.0f}ms")


def write_csv(query_results: list[QueryResult], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "mode", "mrr", "recall_at_5", "latency_ms",
            "top1_section", "top1_type", "top1_url",
        ])
        writer.writeheader()
        for qr in query_results:
            top1 = qr.top_results[0] if qr.top_results else {}
            writer.writerow({
                "query":        qr.query,
                "mode":         qr.mode,
                "mrr":          round(qr.mrr, 4),
                "recall_at_5":  qr.recall_at_5,
                "latency_ms":   round(qr.latency_ms, 1),
                "top1_section": top1.get("section", ""),
                "top1_type":    top1.get("content_type", ""),
                "top1_url":     top1.get("source_url", ""),
            })
    print(f"\nResults written → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test Pipecat retrieval")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run all queries and output CSV")
    parser.add_argument("--output", default="results.csv",
                        help="CSV output path (with --benchmark)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Pre-encode ALL queries in one model.encode() call to avoid macOS
    # PyTorch thread-pool segfaults between repeated encode() invocations.
    import logging
    logging.getLogger().info("Pre-encoding %d queries …", len(TEST_CASES))
    all_queries = [tc.query for tc in TEST_CASES]
    query_vecs  = precompute_query_vectors(all_queries)  # shape: (N, dim)

    query_results: list[QueryResult] = []
    for tc, q_vec in zip(TEST_CASES, query_vecs):
        qr = evaluate(tc, top_k=args.top_k, q_vec=q_vec)
        query_results.append(qr)
        print_result_block(tc, qr)

    print_summary_table(query_results)

    if args.benchmark:
        write_csv(query_results, args.output)


if __name__ == "__main__":
    main()
