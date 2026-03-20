"""
In-process test for the Pipecat MCP server.
No HTTP server needed — FastMCP Client connects directly to the mcp object.

Run:
    python test_server.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import asyncio
from fastmcp import Client
from server import mcp


def _extract_text(result) -> str:
    """Extract plain text from a FastMCP tool result."""
    if isinstance(result, str):
        return result
    # FastMCP returns a list of content objects
    if isinstance(result, list):
        parts = []
        for item in result:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


async def run_tests():
    async with Client(mcp) as client:

        # ── Test 1: General search ────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("TEST 1: search_pipecat_docs")
        print("=" * 72)
        r = await client.call_tool(
            "search_pipecat_docs",
            {"query": "How do I configure Deepgram Nova-3?", "max_results": 3},
        )
        print(_extract_text(r))

        # ── Test 2: Code examples ─────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("TEST 2: get_example_code")
        print("=" * 72)
        r = await client.call_tool(
            "get_example_code",
            {"use_case": "voice bot with OpenAI and Daily transport", "max_examples": 1},
        )
        print(_extract_text(r))

        # ── Test 3: Concept explanation ───────────────────────────────────────
        print("\n" + "=" * 72)
        print("TEST 3: explain_concept")
        print("=" * 72)
        r = await client.call_tool(
            "explain_concept",
            {"concept": "frame"},
        )
        print(_extract_text(r))

        # ── Test 4: Service comparison ────────────────────────────────────────
        print("\n" + "=" * 72)
        print("TEST 4: compare_services")
        print("=" * 72)
        r = await client.call_tool(
            "compare_services",
            {"service_a": "Deepgram", "service_b": "AssemblyAI", "aspect": "setup"},
        )
        print(_extract_text(r))

        print("\n" + "=" * 72)
        print("ALL TESTS PASSED")
        print("=" * 72)


if __name__ == "__main__":
    asyncio.run(run_tests())
