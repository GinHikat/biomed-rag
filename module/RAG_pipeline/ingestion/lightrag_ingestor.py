"""
Ingest biomedical corpora into LightRAG.

Supported sources:
  - BC5CDR   (PubTator format via module.data_processing.bc5cdr)
  - Plain text files / directories

Usage:
    import asyncio
    from module.RAG_pipeline.ingestion.lightrag_ingestor import ingest_bc5cdr

    asyncio.run(ingest_bc5cdr(rag, split="Training"))
"""

import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def _bc5cdr_row_to_text(row) -> str:
    """Format a BC5CDR DataFrame row as plain text for LightRAG insertion."""
    parts = [f"Title: {row['title']}", f"Abstract: {row['abstract']}"]

    # Append entity annotations so the KG builder can extract relations
    chemicals = [e["text"] for e in row["entities"] if e["type"] == "Chemical"]
    diseases  = [e["text"] for e in row["entities"] if e["type"] == "Disease"]
    if chemicals:
        parts.append("Chemicals mentioned: " + ", ".join(sorted(set(chemicals))))
    if diseases:
        parts.append("Diseases mentioned: " + ", ".join(sorted(set(diseases))))

    return "\n".join(parts)


# ── public API ───────────────────────────────────────────────────────────────

async def ingest_bc5cdr(rag, split: str = "Training", batch_size: int = 10):
    """
    Parse BC5CDR and insert all abstracts into a LightRAG instance.

    Args:
        rag:        An initialised LightRAG instance.
        split:      'Training', 'Development', or 'Test'.
        batch_size: Number of documents per ainsert call.
    """
    # Import here so the module works even without the project root on sys.path
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from module.data_processing.bc5cdr import BC5CDR

    parser = BC5CDR()
    df = parser.parse_entity(file_type=split)
    logger.info(f"BC5CDR {split}: loaded {len(df)} documents")

    texts = [_bc5cdr_row_to_text(row) for _, row in df.iterrows()]

    # Insert in batches to avoid overwhelming the LLM server
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Inserting batch {i // batch_size + 1} / {(len(texts) - 1) // batch_size + 1}")
        await rag.ainsert(batch)

    logger.info(f"BC5CDR {split}: ingestion complete ({len(texts)} documents)")


async def ingest_text_files(rag, directory: str, extensions=(".txt",), batch_size: int = 5):
    """
    Recursively read plain-text files from *directory* and insert into LightRAG.

    Args:
        rag:        An initialised LightRAG instance.
        directory:  Root directory to walk.
        extensions: File extensions to include.
        batch_size: Documents per ainsert call.
    """
    paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if any(fname.endswith(ext) for ext in extensions):
                paths.append(os.path.join(root, fname))

    logger.info(f"Found {len(paths)} files in {directory}")

    texts = []
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                texts.append(fh.read())
        except OSError as exc:
            logger.warning(f"Skipping {path}: {exc}")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Inserting file batch {i // batch_size + 1} / {(len(texts) - 1) // batch_size + 1}")
        await rag.ainsert(batch)

    logger.info(f"Text file ingestion complete ({len(texts)} documents)")
