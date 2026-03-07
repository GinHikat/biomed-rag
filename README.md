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

biomedical-rag/

├── config.py                 # Model, paths, entity types

├── preprocess.py             # Multi-source → unified docs

├── index.py                  # Build LightRAG KG + vectors

├── query.py                  # QA interface (CLI)

├── evaluate.py               # RAGAS + CID gold evaluation

├── finetune/                 # (Optional) QLoRA fine-tuning

│   ├── prepare_data.py       # BC5CDR → instruction-tuning format

│   ├── train_qlora.py        # QLoRA training script

│   └── export_ollama.py      # Convert to GGUF → Ollama

├── experiments/              # Ablation & comparison results

│   ├── run_ablation.py       # Compare modes, data sources

│   └── results/              # Saved metrics & plots

├── data/

│   ├── CDR_Data/             # BC5CDR (existing)

│   ├── chemdis_gene/         # ChemDisGene

│   ├── ctd/                  # CTD exports

│   └── processed/            # Unified text docs

├── requirements.txt

└── notebooks/

    └── demo.ipynb

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

This project uses the datasets from the `MedNLPCombined` repository on Hugging Face (`zinzinmit/MedNLPCombined`). You have multiple ways to access and use these datasets for your RAG system.

### Option A: Using the Setup Script (Recommended)
The provided `set_up_dataset.py` automatically pulls the required subsets directly from Hugging Face and structures them correctly into the `data/external/` folder.
By default, this grabs the BioCreative V CDR (`bc5cdr`) and ChemDisGene (`ChemDisGene`) sub-directories.

### Option B: Direct Hugging Face Loading in Python
If you prefer not to download the flat files physically, you can load datasets directly into your Python scripts via the `datasets` or `huggingface_hub` libraries.

#### Using `huggingface_hub` to download specific folders
```python
from huggingface_hub import snapshot_download

# Download only the MedQA and PubMedQA subset text files
snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns=["medqa/**", "pubmedqa/**"],
    local_dir="./data/external"
)
```

#### Accessing raw JSON/TXT files dynamically
Once downloaded via your setup scripts, you can load JSON elements for processing using `json` or Pandas:

```python
import json
import os

# Example: Loading ChemDisGene annotations
file_path = "./data/external/ChemDisGene/data/curated/abstracts.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # process relations directly
        pass
```
