"""
LightRAG pipeline backed by vLLM-served BioMistral + nomic-embed-text.

Quick start:
    import asyncio
    from module.RAG_pipeline.pipeline.rag_pipeline import RAGPipeline

    async def main():
        pipeline = RAGPipeline()
        await pipeline.initialize()
        await pipeline.ingest_bc5cdr()          # one-time indexing
        answer = await pipeline.query("Does aspirin cause gastric bleeding?")
        print(answer)
        await pipeline.close()

    asyncio.run(main())
"""

import asyncio
import logging
import os

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from module.RAG_pipeline.config import (
    WORKING_DIR,
    LLM_MODEL,
    EMBED_DIM,
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    llm_fn,
    embed_fn,
)
from module.RAG_pipeline.ingestion.lightrag_ingestor import ingest_bc5cdr, ingest_text_files

logger = logging.getLogger(__name__)


class RAGPipeline:
    """High-level async RAG pipeline using LightRAG + vLLM."""

    def __init__(self, working_dir: str = WORKING_DIR, mode: str = "hybrid"):
        self.working_dir = working_dir
        self.mode = mode  # 'local' | 'global' | 'hybrid'
        self.rag: LightRAG | None = None

    async def initialize(self):
        """Create and initialise the LightRAG instance."""
        os.makedirs(self.working_dir, exist_ok=True)

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_fn,
            llm_model_name=LLM_MODEL,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBED_DIM,
                max_token_size=EMBED_MAX_TOKENS,
                model_name=EMBED_MODEL,
                func=embed_fn,
            ),
        )
        await self.rag.initialize_storages()
        logger.info("LightRAG initialised (working_dir=%s)", self.working_dir)
        return self

    async def ingest_bc5cdr(self, split: str = "Training", batch_size: int = 10):
        """Index a BC5CDR split into LightRAG."""
        self._assert_ready()
        await ingest_bc5cdr(self.rag, split=split, batch_size=batch_size)

    async def ingest_text_files(self, directory: str, batch_size: int = 5):
        """Index all .txt files under *directory* into LightRAG."""
        self._assert_ready()
        await ingest_text_files(self.rag, directory=directory, batch_size=batch_size)

    async def query(self, question: str, mode: str | None = None) -> str:
        """Answer *question* using LightRAG retrieval."""
        self._assert_ready()
        return await self.rag.aquery(
            question,
            param=QueryParam(mode=mode or self.mode),
        )

    async def close(self):
        """Finalise storages (flush caches, close DB connections)."""
        if self.rag:
            await self.rag.finalize_storages()
            logger.info("LightRAG storages finalised")

    def _assert_ready(self):
        if self.rag is None:
            raise RuntimeError("Call await pipeline.initialize() first.")

    # ── context-manager support ──────────────────────────────────────────────
    async def __aenter__(self):
        return await self.initialize()

    async def __aexit__(self, *_):
        await self.close()
