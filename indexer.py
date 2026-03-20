"""
Build all search indexes for the Pipecat docs retrieval system.

Indexes created:
    data/bm25_index.pkl      – BM25Okapi sparse index
    data/embeddings.npy      – dense embeddings (float32)
    data/faiss_index.bin     – FAISS flat L2 index
    data/chunk_metadata.json – chunk_id → full chunk lookup

Usage:
    python indexer.py --input data/chunks.jsonl --output data/
"""

import os
# Prevent macOS segfault from PyTorch + HuggingFace multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Tokenisation ────────────────────────────────────────────────────────────

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str, preserve_code_terms: bool = False) -> list[str]:
    """
    Tokenize text for BM25.

    For technical/code text, CamelCase identifiers are split so that
    "DeepgramSTTService" also matches "Deepgram", "STT", "Service".
    Stopwords are removed, but short all-caps tokens (acronyms like STT,
    LLM) are always kept.
    """
    tokens: list[str] = []
    for word in text.split():
        word = word.strip(".,;:\"'`()[]{}|\\/<>!?@#$%^&*+=~")
        if not word:
            continue
        # Split CamelCase for technical identifiers
        parts = _CAMEL_RE.split(word) if preserve_code_terms else [word]
        for part in parts:
            lower = part.lower()
            if lower in config.BM25_STOPWORDS and not part.isupper():
                continue
            if lower:
                tokens.append(lower)
                # Also keep the original form for exact matching
                if preserve_code_terms and part != lower:
                    tokens.append(part)
    return tokens if tokens else ["<empty>"]


def chunk_to_bm25_text(chunk: dict[str, Any]) -> str:
    """Concatenate fields for BM25 tokenisation, weighting context fields."""
    parts = []
    # Section hierarchy repeated for weight
    for field in ("section", "h2", "h3"):
        val = chunk.get(field)
        if val:
            parts.extend([val] * 2)
    parts.append(chunk.get("page_title", "") or "")
    if chunk.get("code_language"):
        parts.append(chunk["code_language"])
    parts.append(chunk.get("text", "") or "")
    return " ".join(parts)


def chunk_to_embedding_text(chunk: dict[str, Any]) -> str:
    """Produce a sentence suitable for embedding."""
    ctx_parts = [
        chunk.get("page_title") or "",
        chunk.get("section") or "",
        chunk.get("h2") or "",
        chunk.get("h3") or "",
    ]
    ctx = " > ".join(p for p in ctx_parts if p)
    code_tag = (
        f"[{chunk['code_language']}]" if chunk.get("code_language") else ""
    )
    text = chunk.get("text", "") or ""
    return f"{ctx} {code_tag} {text}".strip()


# ── Index builders ──────────────────────────────────────────────────────────


def build_bm25_index(chunks: list[dict], output_path: Path) -> BM25Okapi:
    """Tokenize all chunks and build a BM25Okapi index."""
    log.info("Building BM25 index for %d chunks …", len(chunks))
    is_code = [c.get("content_type") == "code" for c in chunks]
    tokenized = [
        tokenize(chunk_to_bm25_text(c), preserve_code_terms=code)
        for c, code in tqdm(zip(chunks, is_code), total=len(chunks), desc="Tokenising")
    ]
    bm25 = BM25Okapi(tokenized, k1=config.BM25_K1, b=config.BM25_B)
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    log.info("BM25 index saved → %s", output_path)
    return bm25


def build_dense_embeddings(
    chunks: list[dict], output_path: Path
) -> np.ndarray:
    """
    Embed all chunks with all-MiniLM-L6-v2 and save as float32 numpy array.

    Uses a manual batch loop (no multiprocessing DataLoader) to avoid the
    macOS segfault that occurs when sentence-transformers spawns child processes
    on CPU-only machines.
    """
    log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    texts = [chunk_to_embedding_text(c) for c in chunks]
    log.info("Encoding %d chunks in batches of %d …", len(texts), config.EMBEDDING_BATCH)

    all_embeddings: list[np.ndarray] = []
    batch_size = config.EMBEDDING_BATCH

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[start : start + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,   # outer tqdm handles progress
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeddings.append(vecs.astype(np.float32))

    embeddings = np.vstack(all_embeddings)
    np.save(output_path, embeddings)
    log.info(
        "Embeddings saved → %s  shape=%s  dtype=%s",
        output_path, embeddings.shape, embeddings.dtype,
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray, output_path: Path) -> faiss.Index:
    """Build an exact inner-product FAISS index (works with L2-normalised vectors)."""
    dim = embeddings.shape[1]
    log.info("Building FAISS IndexFlatIP (dim=%d, vectors=%d) …", dim, len(embeddings))
    index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalised vecs
    index.add(embeddings)
    faiss.write_index(index, str(output_path))
    log.info("FAISS index saved → %s  (total=%d)", output_path, index.ntotal)
    return index


def build_metadata_store(chunks: list[dict], output_path: Path) -> dict:
    """
    Build a {chunk_id: chunk} lookup dict and save as JSON.
    Uses a temp file + atomic rename so a crash can't leave a partial file.
    """
    metadata = {c["chunk_id"]: c for c in chunks}
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata, f)
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp_path.replace(output_path)   # atomic on POSIX
    log.info("Metadata store saved → %s  (%d entries)", output_path, len(metadata))
    return metadata


# ── Main ────────────────────────────────────────────────────────────────────


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument(
        "--input", default=str(config.CHUNKS_FILE), help="Path to chunks.jsonl"
    )
    parser.add_argument(
        "--issues", default=str(config.DATA_DIR / "github_issues.jsonl"),
        help="Path to github_issues.jsonl (merged if present)"
    )
    parser.add_argument(
        "--output", default=str(config.DATA_DIR), help="Output directory for indexes"
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    issues_path = Path(args.issues)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    t0 = time.time()
    log.info("Loading chunks from %s …", input_path)
    chunks = load_chunks(input_path)
    log.info("Loaded %d doc chunks", len(chunks))

    if issues_path.exists():
        issue_chunks = load_chunks(issues_path)
        log.info("Merging %d GitHub Issue chunks from %s", len(issue_chunks), issues_path)
        chunks = chunks + issue_chunks
    else:
        log.info("No GitHub Issues file found at %s — skipping (run github_indexer.py to add it)", issues_path)

    log.info("Total chunks to index: %d", len(chunks))

    # Show quick distribution
    from collections import Counter
    type_dist = Counter(c["content_type"] for c in chunks)
    log.info("Content types: %s", dict(type_dist))

    build_bm25_index(chunks, output_dir / "bm25_index.pkl")
    embeddings = build_dense_embeddings(chunks, output_dir / "embeddings.npy")
    build_faiss_index(embeddings, output_dir / "faiss_index.bin")
    build_metadata_store(chunks, output_dir / "chunk_metadata.json")

    elapsed = time.time() - t0
    log.info("All indexes built in %.1fs", elapsed)


if __name__ == "__main__":
    main()
