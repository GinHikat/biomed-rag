"""
LightRAG configuration for vLLM-served BioMistral + nomic-embed-text.

Servers are started by start_lightrag_vllm.sh:
  - LLM  : http://localhost:8080/v1  (BioMistral-7B-AWQ-QGS128-W4-GEMM)
  - Embed : http://localhost:8081/v1  (nomic-ai/nomic-embed-text-v1.5)
"""

import os

import sys
from pathlib import Path

# Add project root and scripts directory to sys.path to import config.py
project_root = Path(__file__).resolve().parent.parent.parent
scripts_dir = project_root / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))
import config

# ── Server endpoints ───────────────────────
LLM_BASE_URL   = getattr(config, "LLM_BASE_URL", "http://localhost:8080/v1")
EMBED_BASE_URL = getattr(config, "EMBED_BASE_URL", "http://localhost:8081/v1")
LLM_API_KEY    = "none"
EMBED_API_KEY  = "none"

LLM_MODEL   = config.LLM_MODEL
EMBED_MODEL = config.EMBEDDING_MODEL
EMBED_DIM   = 768
EMBED_MAX_TOKENS = 512

WORKING_DIR = str(config.RAG_WORKING_DIR)

# ── Async functions passed to LightRAG ───────────────────────────────────────
from lightrag.llm.openai import openai_complete_if_cache, openai_embed


async def llm_fn(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Async LLM completion via vLLM OpenAI-compatible API."""
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        **kwargs,
    )


async def embed_fn(texts):
    """Async embedding via vLLM OpenAI-compatible API."""
    return await openai_embed(
        texts,
        model=EMBED_MODEL,
        base_url=EMBED_BASE_URL,
        api_key=EMBED_API_KEY,
    )
