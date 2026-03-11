"""
Shared RAG setup — imported by ingest.py and query.ipynb.
All tuneable config lives at the top of this file.
"""
import sys, os
from functools import partial

# ── resolve project root (always repo root, regardless of cwd) ────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # notebooks/
project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # repo root
if project_root not in sys.path:
    sys.path.append(project_root)

# ── load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

# ============================================================
# LLM GENERATION CONFIG  (tweak here)
# ============================================================
LLM_MAX_TOKENS  = 3072  # prompt + output must stay under model's 8192-token context window
LLM_TEMPERATURE = 0.1    # lower = more deterministic output
LLM_TOP_P       = 0.95
# ============================================================

# ── debug logging ─────────────────────────────────────────────────────────────
DEBUG_LLM        = True   # set False to disable prompt/response logging
DEBUG_OUTPUT_FILE = os.path.join(project_root, "debug_llm_output.txt")
# ─────────────────────────────────────────────────────────────────────────────

# ── from .env ─────────────────────────────────────────────────────────────────
LLM_MODEL       = os.environ["LLM_MODEL"]
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
LLM_BASE_URL    = os.environ.get("LLM_BASE_URL",  "http://127.0.0.1:8080/v1")
EMBED_BASE_URL   = os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8081/v1")
WORKING_DIR     = os.environ.get("RAG_WORKING_DIR", os.path.join(project_root, "rag_storage"))

# ── lightrag imports ──────────────────────────────────────────────────────────
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, openai_complete

setup_logger("lightrag", level="INFO")

os.makedirs(WORKING_DIR, exist_ok=True)

# ── LLM completion function ───────────────────────────────────────────────────
async def llm_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    """OpenAI-compatible completion routed to the local vLLM server."""
    kwargs.update({
        "temperature": LLM_TEMPERATURE,
        "top_p":       LLM_TOP_P,
        "max_tokens":  LLM_MAX_TOKENS,
    })
    response = await openai_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

    if DEBUG_LLM:
        with open(DEBUG_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt + "\n")
            f.write("=== RESPONSE ===\n")
            f.write(response + "\n")
            f.write("=================\n\n")

    return response

# ── RAG instance ──────────────────────────────────────────────────────────────

# Biomedical entity types — overrides LightRAG's generic defaults
# (default types cause biological entities to be misclassified as 'artifact')
ENTITY_TYPES = [
    "Anatomy",      # organs, tissues, body structures
    "Disease",      # conditions, disorders, syndromes
    "Gene",         # genes, DNA sequences
    "Protein",      # proteins, enzymes, receptors
    "Chemical",     # small molecules, metabolites, ions
    "Drug",         # medications, pharmaceuticals
    "Organism",     # bacteria, viruses, species
    "Procedure",    # medical/surgical procedures, imaging techniques
    "Method",       # laboratory methods, experimental techniques
    "Concept",      # abstract medical/biological concepts
]

def build_rag() -> LightRAG:
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_complete,
        llm_model_name=LLM_MODEL,
        llm_model_max_async=4,
        entity_types=ENTITY_TYPES,
        llm_model_kwargs={
            "base_url": LLM_BASE_URL,
            "api_key":  "none",
        },
        chunk_token_size=1200,
        entity_extract_max_gleaning=0,
        default_embedding_timeout=120,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                base_url=EMBED_BASE_URL,
                api_key="none",
                model=EMBEDDING_MODEL,
            ),
        ),
    )
