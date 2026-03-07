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
├── notebooks/                # Jupyter notebook demonstrations
│   ├── processing_demo/
│   └── rag_demo/
├── experiments/              # Model experiments and ablation studies
├── finetune/                 # Fine-tuning scripts
├── shared_functions/         # Shared utilities (Google Drive/Sheets integration)
├── tests/                    # Unit tests
├── secrets/                  # Credentials directory
├── data/                     # Dataset storage
│   ├── external/             # Hub datasets downloaded via script
│   │   ├── bc5cdr/           # BioCreative V CDR Corpus
│   │   ├── ChemDisGene/      # ChemDisGene dataset
│   │   ├── bioasq/           # BioASQ benchmark dataset
│   │   ├── medqa/            # MedQA multiple-choice dataset
│   │   └── pubmedqa/         # PubMedQA text reasoning dataset
│   └── vectorstore/          # Vector datastore
├── plan.md   
├── todo.txt               
├── set_up_dataset.py         # Dataset preparation script
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variables template

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

