# BioMed RAG — Session Handoff Notes
> Last updated: 2026-03-10  
> Deadline: 2026-03-17 (presentation)

---

## Environment

| Item | Value |
|------|-------|
| Conda env | `pip12` (Python 3.12) |
| `lightrag-hku` | 1.4.10 |
| `vllm` | 0.17.0 |
| `torch` | 2.10.0+cu128 |
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) |

**Install the correct LightRAG package** — there is a name collision on PyPI:
```bash
# ❌ WRONG — installs a completely different SylphAI library
pip install lightrag

# ✅ CORRECT — HKUST LightRAG
pip install lightrag-hku
```

---

## Repository Layout (key files)

```
biomed-rag/
├── start_lightrag_vllm.sh             ← starts all 3 servers (vLLM LLM + embed + LightRAG)
├── module/
│   ├── data_processing/
│   │   ├── bc5cdr.py                  ← BC5CDR PubTator parser (parse_entity, extract_relations)
│   │   ├── ctd.py
│   │   └── pubtator.py
│   └── RAG_pipeline/
│       ├── config.py                  ← llm_fn / embed_fn wrappers pointing at vLLM ports
│       ├── evaluate.py                ← CID F1, RAGAS, MC-QA accuracy
│       ├── ingestion/
│       │   ├── document_loader.py
│       │   └── lightrag_ingestor.py   ← ingest_bc5cdr(), ingest_text_files()
│       └── pipeline/
│           └── rag_pipeline.py        ← RAGPipeline class (initialize/query/close)
├── data/external/
│   ├── bc5cdr/        ✅ downloaded
│   ├── ChemDisGene/   ✅ downloaded
│   ├── bioasq/        ✅ downloaded
│   ├── medqa/         ✅ downloaded
│   └── pubmedqa/      ✅ downloaded
└── notebooks/rag_demo/
    ├── lightrag_biomed_demo.ipynb     ← main demo notebook (end-to-end)
    └── test_light_rag.ipynb           ← earlier scratch notebook
```

---

## Starting the Stack

```bash
cd biomed-rag

# ==========================================
# Option 1: Run all 3 servers in the background
# ==========================================
# Default: both servers on GPU (RTX 3060)
./start_lightrag_vllm.sh

# GPU for embed, CPU for LLM
LLM_DEVICE=cpu ./start_lightrag_vllm.sh

# Both on CPU
LLM_DEVICE=cpu EMBED_DEVICE=cpu ./start_lightrag_vllm.sh


# ==========================================
# Option 2: Run servers individually (better for debugging)
# ==========================================
# Terminal 1: Text Generation (LLM) Server
# You can override LLM_DEVICE=cpu before running
./start_llm_server.sh

# Terminal 2: Embedding Server
# You can override EMBED_DEVICE=cpu before running
./start_embed_server.sh

# Terminal 3: LightRAG Server (waits until both vLLMs are ready)
./start_lightrag_server.sh
```

| Service | Port | URL |
|---------|------|-----|
| vLLM LLM (BioMistral-7B-AWQ) | 8080 | http://localhost:8080/v1 |
| vLLM Embed (nomic-embed-text-v1.5) | 8081 | http://localhost:8081/v1 |
| LightRAG API | 9621 | http://localhost:9621/docs |

Logs are written to `biomed-rag/logs/`.

### Key env vars (all have defaults, override as needed)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_DEVICE` | `gpu` | `gpu` or `cpu` for the LLM server |
| `EMBED_DEVICE` | `gpu` | `gpu` or `cpu` for the embed server |
| `VLLM_GPU_MEM_UTIL` | `0.70` | GPU fraction for LLM (RTX 3060 needs ≤ 0.70) |
| `VLLM_EMBED_GPU_MEM_UTIL` | `0.15` | GPU fraction for embed model |
| `VLLM_QUANTIZATION` | `awq_marlin` | AWQ-Marlin is faster than plain AWQ on RTX |
| `LLM_MODEL` | `BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM` | HuggingFace model ID |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |

---

## Known Issues Fixed

### 1. PyPI name collision
`pip install lightrag` installs `lightrag==0.1.0b6` (SylphAI) which has a completely different API.  
**Fix:** use `pip install lightrag-hku`.

### 2. vLLM OOM on startup
Default `--gpu-memory-utilization 0.9` required 10.47 GiB but only 9.05 GiB was free.  
**Fix:** set `VLLM_GPU_MEM_UTIL=0.70` and `VLLM_EMBED_GPU_MEM_UTIL=0.15`.

### 3. `--task embedding` flag removed in vLLM 0.17.0
**Fix:** replaced with `--runner pooling`.

### 4. `--device cpu` flag not in vLLM 0.17.0
**Fix:** use `CUDA_VISIBLE_DEVICES=""` env var + `--dtype float32` (no quantization) for CPU mode.

### 5. AWQ quantization warning
vLLM recommended `awq_marlin` over `awq` for faster inference on compatible hardware.  
**Fix:** default quantization changed to `awq_marlin`.

---

## What's Built (RAG Pipeline)

### `module/RAG_pipeline/config.py`
Defines `llm_fn` and `embed_fn` async functions that call vLLM via OpenAI-compatible HTTP.  
All endpoints/keys read from env vars so the same code works with the shell script exports.

### `module/RAG_pipeline/pipeline/rag_pipeline.py`
```python
from module.RAG_pipeline.pipeline.rag_pipeline import RAGPipeline

async with RAGPipeline(mode="hybrid") as pipeline:
    await pipeline.ingest_bc5cdr(split="Training")
    answer = await pipeline.query("Does aspirin cause gastric bleeding?")
```

### `module/RAG_pipeline/ingestion/lightrag_ingestor.py`
- `ingest_bc5cdr(rag, split, batch_size)` — parses BC5CDR via `bc5cdr.py`, formats title + abstract + entity annotations, inserts in batches
- `ingest_text_files(rag, directory, extensions, batch_size)` — walks a directory and inserts `.txt` files

### `module/RAG_pipeline/evaluate.py`
| Function | Purpose |
|----------|---------|
| `evaluate_cid_f1(pipeline, split, max_pairs)` | CID relation extraction Precision/Recall/F1 vs BC5CDR gold |
| `evaluate_ragas(pipeline, qa_pairs)` | RAGAS: faithfulness, answer_relevancy, context_precision, context_recall |
| `evaluate_mcqa(pipeline, items, rag_enabled)` | MC-QA accuracy (MedQA style); set `rag_enabled=False` for raw-model baseline |

---

## What Still Needs to Be Done

- [ ] **Run ingestion** — BC5CDR Training split into LightRAG (one-time, ~30 min on GPU)
- [ ] **Collect QA test pairs** — from BioASQ / PubMedQA for RAGAS evaluation (data team)
- [ ] **Run CID F1 eval** — `evaluate_cid_f1(pipeline, split="Test")`
- [ ] **Run RAGAS metrics** — once QA pairs are ready
- [ ] **RAG vs raw-model benchmark** — `evaluate_mcqa(..., rag_enabled=False)` then `rag_enabled=True` on MedQA test set
- [ ] **Ablation study** — compare `local` vs `global` vs `hybrid` retrieval modes
- [ ] **Ingest ChemDisGene** — Phase 2 of the plan (larger scale)

---

## Demo Notebook

Open `notebooks/rag_demo/lightrag_biomed_demo.ipynb`.  
Run cells in order: environment check → init pipeline → ingest → queries → CID F1 → RAGAS.  
The notebook uses `await` syntax — it expects an IPython/Jupyter kernel (not a plain Python script).
