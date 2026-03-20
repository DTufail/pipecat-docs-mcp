"""
Hybrid search retrieval for Pipecat documentation.

Implements:
  - BM25 sparse retrieval
  - Dense semantic retrieval via FAISS
  - RRF (Reciprocal Rank Fusion) merge
  - Optional cross-encoder reranking
  - Query-mode boosting (code, semantic, keyword)

Usage:
    python retrieval.py --query "How do I use Deepgram?" --mode hybrid --top-k 5
"""

import os
# Must be set BEFORE torch / tokenizers are imported to prevent macOS segfault
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import argparse
import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from indexer import tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Loader ───────────────────────────────────────────────────────────────────

class IndexStore:
    """Lazily loads all indexes and keeps them in memory."""

    def __init__(self, data_dir: Path = config.DATA_DIR):
        self._data_dir = data_dir
        self._bm25       = None
        self._tokenized  = None
        self._embedder   = None
        self._faiss      = None
        self._metadata   = None
        self._reranker   = None

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def bm25(self):
        if self._bm25 is None:
            path = self._data_dir / "bm25_index.pkl"
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self._bm25      = obj["bm25"]
            self._tokenized = obj["tokenized"]
            log.info("BM25 index loaded (%d docs)", len(self._tokenized))
        return self._bm25, self._tokenized

    # ── Embedder ─────────────────────────────────────────────────────────────

    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
            self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._embedder

    # ── FAISS ────────────────────────────────────────────────────────────────

    def faiss_index(self) -> faiss.Index:
        if self._faiss is None:
            path = self._data_dir / "faiss_index.bin"
            self._faiss = faiss.read_index(str(path))
            log.info("FAISS index loaded (%d vectors)", self._faiss.ntotal)
        return self._faiss

    # ── Metadata ─────────────────────────────────────────────────────────────

    def metadata(self) -> dict[str, dict]:
        if self._metadata is None:
            path = self._data_dir / "chunk_metadata.json"
            with open(path) as f:
                self._metadata = json.load(f)
            log.info("Metadata loaded (%d chunks)", len(self._metadata))
        return self._metadata

    # ── Reranker (optional) ───────────────────────────────────────────────────

    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading reranker: %s", config.RERANKER_MODEL)
            self._reranker = CrossEncoder(config.RERANKER_MODEL)
        return self._reranker

    def preload_all(self) -> None:
        """Load every index into memory upfront."""
        self.bm25()
        self.embedder()
        self.faiss_index()
        self.metadata()


# ── Module-level singleton ─────────────────────────────────────────────────

_store: Optional[IndexStore] = None


def get_store() -> IndexStore:
    global _store
    if _store is None:
        _store = IndexStore()
    return _store


# ── Query helpers ────────────────────────────────────────────────────────────

_CODE_HINTS    = re.compile(r"\b(example|show|code|snippet|how to|implement)\b", re.I)
_CONCEPT_HINTS = re.compile(r"\b(what is|explain|understand|concept|difference|why)\b", re.I)
_ERROR_HINTS   = re.compile(r"(Error|Exception|traceback|fix|debug|issue|problem)", re.I)


def detect_query_intent(query: str) -> str:
    """Return 'code', 'concept', 'error', or 'general'."""
    if _ERROR_HINTS.search(query):
        return "error"
    if _CODE_HINTS.search(query):
        return "code"
    if _CONCEPT_HINTS.search(query):
        return "concept"
    return "general"


# ── Core retrieval ───────────────────────────────────────────────────────────

def _bm25_search(query: str, top_n: int = 20) -> list[tuple[int, float]]:
    """Return (doc_index, score) pairs from BM25."""
    bm25, _ = get_store().bm25()
    tokens = tokenize(query, preserve_code_terms=True)
    scores = bm25.get_scores(tokens)
    # argsort descending, take top_n
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(int(i), float(scores[i])) for i in top_idx]


def precompute_query_vectors(queries: list[str]) -> np.ndarray:
    """
    Encode a list of queries in one shot.
    Call this once before a batch of searches to avoid repeated model.encode()
    calls (which cause macOS PyTorch thread-pool crashes between invocations).
    Returns float32 array of shape (len(queries), embedding_dim).
    """
    model = get_store().embedder()
    return model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def _dense_search(
    query: str,
    top_n: int = 20,
    q_vec: "np.ndarray | None" = None,
) -> list[tuple[int, float]]:
    """Return (doc_index, score) pairs from FAISS inner product search.

    Parameters
    ----------
    q_vec : optional pre-computed query embedding (shape 1×dim).
            If provided, skips model.encode() entirely.
    """
    index = get_store().faiss_index()
    if q_vec is None:
        model = get_store().embedder()
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec.reshape(1, -1), top_n)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


def _rrf_merge(
    bm25_results:  list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    k: int = config.RRF_K,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    score(d) = Σ  1 / (k + rank_i(d))
    """
    rrf: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(dense_results, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def _apply_mode_boost(
    ranked: list[tuple[int, float]],
    mode: str,
    chunks_list: list[dict],
) -> list[tuple[int, float]]:
    """Boost code or text chunks based on search mode."""
    if mode not in ("code", "semantic"):
        return ranked
    boosted = []
    for idx, score in ranked:
        ct = chunks_list[idx].get("content_type", "text")
        if mode == "code" and ct == "code":
            score *= config.CODE_BOOST
        elif mode == "semantic" and ct == "text":
            score *= 1.5
        boosted.append((idx, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


def _index_to_chunk(doc_idx: int) -> dict:
    """Look up a chunk by its positional index in the metadata store."""
    # metadata is keyed by chunk_id; build positional lookup on first call
    store = get_store()
    meta  = store.metadata()
    if not hasattr(store, "_id_list"):
        # Build ordered list matching the order indexer.py used:
        # 1) chunks.jsonl  2) github_issues.jsonl (if present)
        store._id_list = []
        sources = [config.CHUNKS_FILE, config.DATA_DIR / "github_issues.jsonl"]
        for path in sources:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        store._id_list.append(json.loads(line)["chunk_id"])
    chunk_id = store._id_list[doc_idx]
    return meta[chunk_id]


# ── Public API ───────────────────────────────────────────────────────────────

def search(
    query: str,
    mode: str = "hybrid",
    top_k: int = config.DEFAULT_TOP_K,
    rerank: bool = False,
    q_vec: "np.ndarray | None" = None,
) -> list[dict]:
    """
    Search the Pipecat documentation.

    Parameters
    ----------
    query : str
        Natural language or code-focused query.
    mode : str
        'hybrid'   – BM25 + dense + RRF (default)
        'semantic' – dense only
        'keyword'  – BM25 only
        'code'     – hybrid with 3× boost on code chunks
    top_k : int
        Number of results to return.
    rerank : bool
        Apply cross-encoder reranking (slower, more precise).

    Returns
    -------
    List of dicts with keys: chunk_id, text, source_url, section,
    h2, h3, content_type, code_language, score, retrieval_method.
    """
    if not query.strip():
        return []

    t0 = time.perf_counter()

    # Retrieve candidates
    pool = max(top_k * 4, config.BM25_CANDIDATES)
    bm25_res, dense_res = [], []

    if mode in ("hybrid", "keyword", "code"):
        bm25_res = _bm25_search(query, top_n=pool)

    if mode in ("hybrid", "semantic", "code"):
        dense_res = _dense_search(query, top_n=pool, q_vec=q_vec)

    # Merge
    if mode == "keyword":
        ranked = [(i, s) for i, s in bm25_res]
    elif mode == "semantic":
        ranked = [(i, s) for i, s in dense_res]
    else:
        ranked = _rrf_merge(bm25_res, dense_res)

    # Mode-based boosting
    # Need a list of chunks by positional index for boosting
    # We'll resolve lazily below; for boosting we only need content_type
    if mode in ("code", "semantic"):
        ranked = _apply_mode_boost(ranked, mode, _get_chunks_list())

    # Resolve to chunk dicts
    results = []
    for idx, score in ranked[: top_k * 2]:   # fetch 2× then trim after rerank
        chunk = _index_to_chunk(idx)
        results.append({
            **chunk,
            "score": round(score, 6),
            "retrieval_method": mode,
        })

    # Rerank (optional)
    if rerank and results:
        results = _rerank(query, results, top_k=top_k)
    else:
        results = results[:top_k]

    latency_ms = (time.perf_counter() - t0) * 1000
    log.debug("search(%r, mode=%s, top_k=%d) → %.1fms", query, mode, top_k, latency_ms)

    return results


def _get_chunks_list() -> list[dict]:
    """Return all chunks as a positional list (cached on store).
    Must match the order indexer.py used: chunks.jsonl then github_issues.jsonl."""
    store = get_store()
    if not hasattr(store, "_chunks_list"):
        import json as _json
        store._chunks_list = []
        sources = [config.CHUNKS_FILE, config.DATA_DIR / "github_issues.jsonl"]
        for path in sources:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        store._chunks_list.append(_json.loads(line))
    return store._chunks_list


def _rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Cross-encoder reranking of candidate chunks."""
    reranker = get_store().reranker()
    pairs    = [(query, c["text"]) for c in candidates]
    scores   = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    for c in candidates:
        c["retrieval_method"] += "+rerank"
    return candidates[:top_k]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_result(rank: int, r: dict) -> None:
    breadcrumb = " > ".join(
        p for p in [r.get("section"), r.get("h2"), r.get("h3")] if p
    )
    lang = f" [{r['code_language']}]" if r.get("code_language") else ""
    score = r.get("rerank_score", r["score"])
    print(f"\n  {rank}. [{score:.4f}] {breadcrumb}{lang}")
    print(f"     {r['source_url']}")
    preview = r["text"][:240].replace("\n", " ")
    print(f"     {preview}…" if len(r["text"]) > 240 else f"     {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Pipecat documentation")
    parser.add_argument("--query",  required=True)
    parser.add_argument("--mode",   default="hybrid",
                        choices=["hybrid", "semantic", "keyword", "code"])
    parser.add_argument("--top-k",  type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    results = search(args.query, mode=args.mode, top_k=args.top_k, rerank=args.rerank)
    print(f'\nQuery: "{args.query}"  mode={args.mode}  top_k={args.top_k}')
    for i, r in enumerate(results, 1):
        _print_result(i, r)
    print()


if __name__ == "__main__":
    main()
