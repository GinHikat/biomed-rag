import json
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = os.getcwd()

if not (BASE_DIR.endswith("biomed-rag")):
    raise ValueError("Please run this script from the nlp-project or biomed-rag directory")

data_dir = os.path.join(BASE_DIR, "data")

os.makedirs(data_dir, exist_ok=True)

external_data_dir = os.path.join(data_dir, "external")
os.makedirs(external_data_dir, exist_ok=True)

vectorstore_data_dir = os.path.join(data_dir, "vectorstore")
os.makedirs(vectorstore_data_dir, exist_ok=True)

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="bc5cdr/**",
    local_dir=external_data_dir
)

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="ChemDisGene/**",
    local_dir=external_data_dir
)

# --- 5. Reorganize pqaa & pqau into subdirectories ---
file_moves = [
    ("pqaa_train_set.json", "pqaa/train_set.json"),
    ("pqaa_dev_set.json", "pqaa/dev_set.json"),
    ("ori_pqau.json", "pqau/pqau.json"),
]

for old, new in file_moves:
    src = pubmed_qa_dir / old
    dst = pubmed_qa_dir / new
    if src.exists():
        dst.parent.mkdir(exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"Moved {old} -> {new}")
