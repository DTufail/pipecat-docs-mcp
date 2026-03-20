# pipecat-docs-mcp

An MCP server that makes all 317 pages of Pipecat documentation searchable from Claude Desktop and Cursor.

## Why this exists

Pipecat is a fast-moving voice AI framework with documentation spread across STT providers, TTS providers, transports, pipeline architecture, and deployment guides. Finding the right configuration snippet — or understanding how a `Frame` flows through a `Pipeline` — means jumping between five different sections, reading pages that don't quite apply, and losing the thread of what you were building. This server indexes all 6,936 documentation chunks and exposes them through four focused search tools, so Claude can answer Pipecat questions with grounded, source-cited results instead of hallucinating API signatures.


## Performance

| Metric | Value |
|---|---|
| Pages scraped | 317 / 317 (0 errors) |
| Chunks indexed | 6,936 |
| Mean Reciprocal Rank | 0.788 |
| Recall@5 | 93% |
| Average query latency | 36ms |
| Index RAM footprint | ~76MB |

## Architecture

The retrieval pipeline runs two searches in parallel — BM25 for keyword precision and a dense vector search using `all-MiniLM-L6-v2` embeddings for semantic similarity — then merges them with Reciprocal Rank Fusion. The fused candidates are reranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) that scores query-chunk relevance directly, yielding a final ordered list. Chunking respects h2/h3 section boundaries from the original docs and keeps code blocks intact so retrieved code is always runnable.

```
Query → [BM25 top-20]   ──┐
                           ├── RRF Fusion → [Cross-Encoder Reranker] → Top-K Results
Query → [Dense top-20]  ──┘
```

## Tools

### `search_pipecat_docs`

General-purpose search across all documentation — use for configuration questions, error messages, and how-to queries.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Natural language question or keyword |
| `max_results` | `int` | `5` | Number of results to return (max 10) |
| `mode` | `str` | `"hybrid"` | Search mode: `hybrid`, `semantic`, `keyword`, or `code` |

```
User: How do I configure Deepgram Nova-3 with smart_format?
Claude calls: search_pipecat_docs(query="Deepgram Nova-3 smart_format configuration", max_results=3)
```

---

### `get_example_code`

Returns complete, runnable Python pipeline examples filtered to code chunks only — use when you need working code, not documentation prose.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_case` | `str` | required | Description of the pipeline (e.g. "voice bot with Deepgram and OpenAI") |
| `max_examples` | `int` | `2` | Number of examples to return |

```
User: Show me a complete voice pipeline using Daily transport and OpenAI.
Claude calls: get_example_code(use_case="voice bot with Daily transport and OpenAI", max_examples=2)
```

---

### `explain_concept`

Explains Pipecat architecture concepts by pulling from foundational documentation sections — use for "what is X" and "how does X work" questions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `concept` | `str` | required | Concept name (e.g. `frame`, `pipeline`, `VAD`, `transport`) |

```
User: What is a Frame in Pipecat?
Claude calls: explain_concept(concept="frame")
```

---

### `compare_services`

Returns side-by-side documentation for two Pipecat-compatible providers — use when choosing between STT, TTS, or transport options.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `service_a` | `str` | required | First service name (e.g. `Deepgram`) |
| `service_b` | `str` | required | Second service name (e.g. `AssemblyAI`) |
| `aspect` | `str` | `""` | Optional focus: `latency`, `cost`, `setup`, `features` |

```
User: Compare Deepgram and AssemblyAI for setup complexity.
Claude calls: compare_services(service_a="Deepgram", service_b="AssemblyAI", aspect="setup")
```

## Installation

```bash
git clone https://github.com/your-username/pipecat-docs-mcp.git
cd pipecat-docs-mcp
pip install -r requirements.txt
```

Build the search index (the scraped `chunks.jsonl` is included — this only builds the BM25/FAISS indexes):

```bash
python indexer.py
```

This takes about 2 minutes and writes ~30MB of index files to `data/`.

## Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pipecat-docs": {
      "command": "/absolute/path/to/your/venv/bin/python3",
      "args": ["/absolute/path/to/pipecat-docs-mcp/server.py"]
    }
  }
}
```

Use the Python binary from your virtual environment, not system Python, to ensure all dependencies are on the path. Restart Claude Desktop after saving.

## Cursor / Other MCP Clients

Start the HTTP server:

```bash
fastmcp run server.py:mcp --transport http --port 8000
```

Then connect your client to `http://localhost:8000/mcp`.

## Project Structure

```
pipecat-docs-mcp/
├── server.py              # FastMCP server — 4 MCP tools
├── retrieval.py           # Hybrid search engine (BM25 + FAISS + RRF + reranker)
├── indexer.py             # Builds BM25, FAISS, and metadata indexes
├── test_server.py         # In-process tests for all 4 tools
├── test_retrieval.py      # Retrieval evaluation — MRR, Recall@5, latency
├── requirements.txt
├── pipecat-scraper/
│   ├── scraper.py         # Sitemap-driven scraper (resumable, 317 pages)
│   ├── utils.py           # HTML parsing and chunking logic
│   ├── audit.py           # Per-page extraction audit script
│   └── config.py
└── data/
    ├── chunks.jsonl        # 6,936 scraped chunks
    ├── bm25_index.pkl      # BM25 sparse index
    ├── faiss_index.bin     # FAISS dense index
    ├── embeddings.npy      # 384-dim float32 vectors (6936 × 384)
    └── chunk_metadata.json # Chunk metadata (title, section, URL, type)
```

## Roadmap

- **Error-driven retrieval** — index GitHub Issues and Discussions so Claude can match runtime errors to reported bugs and community fixes.
- **Auto-refresh on release** — trigger re-scrape and re-index on new Pipecat releases via GitHub webhook or cron.
- **Expanded coverage** — add LiveKit and Vapi documentation as additional indexed sources.

## License

MIT
