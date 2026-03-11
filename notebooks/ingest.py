"""
ingest.py — Insert documents into the LightRAG knowledge graph.

Usage:
    python ingest.py                        # ingest default textbook
    python ingest.py path/to/your/file.txt  # ingest a custom file
"""
import asyncio
import sys

from rag_config import build_rag, project_root, LLM_MODEL, EMBEDDING_MODEL, LLM_BASE_URL, WORKING_DIR, LLM_MAX_TOKENS

# ── Document to ingest ────────────────────────────────────────────────────────
import os
DEFAULT_TEXTBOOK = os.path.join(
    project_root, "data", "external", "medqa", "textbooks", "Anatomy_Gray.txt"
)

async def main():
    source_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXTBOOK

    print(f"  LLM model    : {LLM_MODEL}")
    print(f"  Embed model  : {EMBEDDING_MODEL}")
    print(f"  LLM base URL : {LLM_BASE_URL}")
    print(f"  Working dir  : {WORKING_DIR}")
    print(f"  max_tokens   : {LLM_MAX_TOKENS}")
    print(f"  Source file  : {source_file}")
    print()

    rag = build_rag()
    await rag.initialize_storages()

    with open(source_file, "r") as f:
        text = f.read()

    print(f"Inserting {len(text):,} chars from '{os.path.basename(source_file)}' ...")
    await rag.ainsert(text)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
