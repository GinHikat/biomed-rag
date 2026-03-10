#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import torch
from functools import partial
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import sys, os
project_root = '/home/zendragonxxx/Programming/tung-nlp/biomed-rag'

if not project_root.endswith("biomed-rag"):
    raise ValueError(f"Project root {project_root} does not end with 'biomed-rag'. Please check the path.")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# lightrag_root = os.path.abspath(os.path.join(os.getcwd(), "../../../LightRAG"))
# if lightrag_root not in sys.path:
#     sys.path.insert(0, lightrag_root)

# import asyncio
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.hf import hf_embed
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.utils import setup_logger


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# In[ ]:


setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


async def hf_model_complete(prompt: str, **kwargs):

    inputs = tokenizer(prompt, return_tensors="pt")

    # remove token_type_ids if present
    inputs.pop("token_type_ids", None)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------
# Embedding model
# -------------------------

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


async def initialize_rag():

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name=MODEL_NAME,

        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=2048,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            func=partial(
                hf_embed,
                embed_model=embed_model,
                tokenizer = tokenizer
            )
        ),
    )

    await rag.initialize_storages()
    return rag


async def main():

    rag = None

    try:

        rag = await initialize_rag()

        await rag.ainsert("""
        Proteins are the most abundant and functionally diverse molecules in living systems.
        Virtually every life process depends on this class of macromolecules.
        Enzymes regulate metabolism, contractile proteins permit movement,
        and proteins such as hemoglobin transport oxygen in the bloodstream.
        """)

        result = await rag.aquery(
            "What are the main functions of proteins?",
            param=QueryParam(mode="hybrid")
        )

        print(result)

    except Exception as e:
        print("Error:", e)

    finally:
        if rag:
            await rag.finalize_storages()

import asyncio
asyncio.run(main())


# In[ ]:




