import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SECRETS_DIR = REPO_ROOT / "secrets"

# Load environment variables from .env
load_dotenv(REPO_ROOT / ".env")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "none")

# LLM Configuration (Safe to push)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
VLLM_QUANTIZATION = "awq_marlin"
LLM_BASE_URL = "http://127.0.0.1:8080/v1"

# Embedding Configuration (Safe to push)
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_BASE_URL = "http://127.0.0.1:8081/v1"
EMBEDDING_DIM = 768
EMBEDDING_MAX_TOKENS = 512

# RAG Configuration
RAG_WORKING_DIR = REPO_ROOT / "rag_storage"

# Google API Configuration (Read from .env)
GOOGLE_API_CREDS = os.getenv("GOOGLE_API_CREDS", str(SECRETS_DIR / "ggsheet_credentials.json"))
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_DRIVE_ID = os.getenv("GOOGLE_DRIVE_ID")

# Default Server Settings
VLLM_HOST = "0.0.0.0"
VLLM_LLM_PORT = 8080
VLLM_EMBED_PORT = 8081
LIGHTRAG_HOST = "0.0.0.0"
LIGHTRAG_PORT = 9621
