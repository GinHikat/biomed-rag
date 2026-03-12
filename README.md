This repo is created for the Final Project Presentation of the NLP course in USTH

Members:

- Pham Quang Vinh
- Hoang Khanh Dong
- Nguyen Lam Tung
- Pham Duy Anh
- Nguyen Vu Hong Ngoc
- Le Chi Thanh Lam

## Project Structure

```

biomed-rag/
├── module/                   # Main project source code
│   ├── RAG_pipeline/         # Core RAG components (chunking, embeddings, generation, ingestion, pipeline, retrieval, vectorstore)
│   └── data_processing/      # Dataset parsing scripts (bc5cdr.py, ctd.py, pubtator.py)
├── notebooks/                # Jupyter notebook demonstrations and RAG scripts
│   ├── ingest.py             # Single file ingestion
│   ├── ingest_full.py        # Full corpus ingestion
│   └── rag_config.py         # Shared RAG configuration
├── scripts/                  # Infrastructure and server startup scripts
│   ├── config.py             # Centralized infrastructure configuration
│   ├── start_llm_server.py   # vLLM LLM server launcher
│   ├── start_embed_server.py # vLLM embedding server launcher
│   └── start_lightrag_server.py # LightRAG server launcher
├── shared_functions/         # Shared utilities (Google Drive/Sheets integration)
├── data/                     # Dataset storage
├── secrets/                  # Credentials directory
├── requirements.txt          # Python dependencies
└── .env                      # Secrets (Google API, etc. - do not push)
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/PhQuangVinh2005/biomed-rag.git
cd biomed-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the template and fill in your details (credentials for Google Sheets and local environment variables).

```bash
cp .env.example .env
```

### 4. Data Preparation

Run the setup script to download and structure the dataset:

```bash
python set_up_dataset.py
```

## Using Datasets (MedNLPCombined)

This project integrates various datasets from the `MedNLPCombined` repository on Hugging Face (`zinzinmit/MedNLPCombined`). The provided `set_up_dataset.py` script automatically downloads these datasets into `data/external/`.

Here is a summary of the available datasets:

| Dataset               | Introduction                                                         | Purpose                                                                     | Format (Original))      | Size (approx.)                             |
| --------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------- | ------------------------------------------ |
| **BC5CDR**      | BioCreative V Task Corpus for Chemical-Disease Relations.            | Named Entity Recognition (NER) and Relation Extraction (RE) for CID.        | `PubTator` / `.txt` | 1,500 annotated PubMed articles            |
| **ChemDisGene** | Large-scale distant-supervision relationship corpus.                 | Extracting biomedical relationships between chemicals, diseases, and genes. | `.txt` / `.tsv`     | ~80,000 biomedical abstracts               |
| **BioASQ**      | Benchmark dataset for biomedical semantic indexing and QA.           | Evaluating RAG systems on factoid, list, and boolean questions.             | `.json` / `.txt`    | Thousands of QA pairs & reference contexts |
| **MedQA**       | Medical multiple-choice question answering dataset from board exams. | Testing comprehensive clinical reasoning and domain professional knowledge. | `.jsonl`              | ~61,000 multiple-choice questions          |
| **PubMedQA**    | QA dataset requiring model reasoning over scientific abstracts.      | Answering research questions based on PubMed abstracts with yes/no/maybe.   | `.json`               | ~273,000 context-question pairs            |

### Temporary Note for the dataset

The temporary document corpus for RAG can be found in MedQA dataset or by aggregating the Title/Abstract column in BC5CDR and ChemDisGene

```bash
data/medqa/textbooks

data/bc5cdr/data/training/full_bc5cdr_data.csv

data/ChemDisGene/data/main/ctd_full_data
```

## Running LightRAG

Follow these steps to set up and run the biomedical RAG system:

### 1. Start vLLM Servers
First, launch the text generation and embedding servers using the scripts in the `scripts/` directory.

**Terminal 1: LLM Server**
```bash
python scripts/start_llm_server.py
```

**Terminal 2: Embedding Server**
```bash
python scripts/start_embed_server.py
```

### 2. Ingest Documents
Once both servers are running, you can ingest the biomedical corpus into the vector store.

```bash
# Ingest all available datasets (MedQA textbooks, PubMedQA, etc.)
python notebooks/ingest_full.py

# OR ingest a specific source (default Gray's Anatomy)
python notebooks/ingest.py
```

### 3. Start LightRAG Server & WebUI
Finally, start the LightRAG server which provides the API and Web interface.

```bash
python scripts/start_lightrag_server.py
```

Open your browser and navigate to:
**[http://localhost:9621](http://localhost:9621)** (Default LightRAG port)

