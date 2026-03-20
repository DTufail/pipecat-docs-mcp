"""Configuration and constants for pipecat-mcp retrieval system."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).parent
DATA_DIR         = ROOT_DIR / "data"
CHUNKS_FILE      = DATA_DIR / "chunks.jsonl"
BM25_INDEX_FILE  = DATA_DIR / "bm25_index.pkl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
EMBEDDINGS_FILE  = DATA_DIR / "embeddings.npy"
METADATA_FILE    = DATA_DIR / "chunk_metadata.json"

# ── Embedding model ────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384          # all-MiniLM-L6-v2 output dim
EMBEDDING_BATCH    = 32           # batch size during indexing (lower = safer on macOS CPU)

# ── Reranker model (optional) ──────────────────────────────────────────────
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── BM25 hyperparameters ───────────────────────────────────────────────────
BM25_K1 = 1.5   # term frequency saturation
BM25_B  = 0.75  # length normalization

# ── RRF hyperparameters ────────────────────────────────────────────────────
RRF_K           = 60   # rank fusion constant
BM25_CANDIDATES = 20   # sparse retrieval pool
DENSE_CANDIDATES = 20  # dense retrieval pool

# ── Query defaults ─────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
CODE_BOOST    = 3.0   # score multiplier when mode='code'

# ── Stopwords to strip from BM25 (but preserve technical terms) ────────────
BM25_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "this", "that", "these", "those",
}
