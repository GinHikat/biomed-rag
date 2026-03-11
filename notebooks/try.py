# %%
# %%

# ============================================================
# LLM GENERATION CONFIG  (change these to tune extraction)
# ============================================================
LLM_MAX_TOKENS  = 2048   # increase if relations are cut off mid-list
LLM_TEMPERATURE = 0.1    # lower = more deterministic output
LLM_TOP_P       = 0.95
# ============================================================
import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt 
# import seaborn as sns
import torch
from functools import partial
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.hf import hf_embed
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed, openai_complete
from lightrag.utils import setup_logger

# %%
setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# %%
async def hf_model_complete(prompt: str, system_prompt=None, history_messages=[], **kwargs):
    device = model.device
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Use chat template to wrap the complex LightRAG instructions
    tokenized_chat = llm_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inputs = {"input_ids": tokenized_chat.to(device)}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )

    decoded = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return decoded.strip()

# %%
async def bio_mistral_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    # Update kwargs to be more strict for extraction
    kwargs.update({
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOKENS,
    })

    import re
    
    # LightRAG sends a massive prompt with its own instructions and examples.
    # We need to extract JUST the raw text to be processed.
    text_to_process = prompt
    match = re.search(r'---Data to be Processed---.*?<Input Text>\n*```(.*?)```', prompt, re.DOTALL)
    if match:
        text_to_process = match.group(1).strip()
    
    # Completely override the prompt to guarantee BioMistral doesn't get confused by LightRAG's template
    custom_system_prompt = (
        "You are an expert biomedical knowledge graph extractor. Your task is to extract medical entities and relationships from the provided text.\n\n"
        "Guidelines:\n"
        "- Extract entities that are Anatomy, Disease, Gene, Chemical, Procedure, or Concept.\n"
        "- Do not extract information that is not explicitly in the text.\n"
        "- Output exactly 4 fields for entities, separated by <|#|>.\n"
        "- Output exactly 5 fields for relationships, separated by <|#|>.\n"
        "- Output ONLY one item per line. End with <|COMPLETE|>.\n\n"
        "EXAMPLE INPUT TEXT:\n"
        "Aspirin is often used to treat mild headaches. It works by inhibiting cyclooxygenase.\n\n"
        "EXAMPLE EXPECTED OUTPUT:\n"
        "entity<|#|>Aspirin<|#|>Chemical<|#|>Medication used to treat pain and inhibit cyclooxygenase.\n"
        "entity<|#|>Headache<|#|>Disease<|#|>Condition treated by aspirin.\n"
        "entity<|#|>Cyclooxygenase<|#|>Gene<|#|>Enzyme inhibited by aspirin.\n"
        "relation<|#|>Aspirin<|#|>Headache<|#|>treats<|#|>Aspirin provides relief for headaches.\n"
        "relation<|#|>Aspirin<|#|>Cyclooxygenase<|#|>inhibits<|#|>Aspirin inhibits the activity of cyclooxygenase.\n"
        "<|COMPLETE|>\n\n"
        "Now, process the following real input text and produce ONLY the extraction list."
    )
    
    final_prompt = f"{custom_system_prompt}\n\nREAL INPUT TEXT:\n{text_to_process}\n\nREAL EXTRACTION OUTPUT:\n"

    response = await openai_complete(
        final_prompt,
        system_prompt=None, # Already prepended
        history_messages=history_messages,
        **kwargs
    )
    
    # Fix BioMistral single-line issue
    response = response.replace("entity<|#|>", "\nentity<|#|>")
    response = response.replace("relation<|#|>", "\nrelation<|#|>")
    response = response.strip()

    # Diagnostic logging
    with open("debug_llm_output.txt", "a") as f:
        f.write("=== PROMPT ===\n")
        f.write(prompt + "\n")
        f.write("=== RESPONSE ===\n")
        f.write(response + "\n")
        f.write("=================\n\n")
        
    return response

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, 'biomed-rag', '.env'))

LLM_MODEL = os.environ["LLM_MODEL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

# %%
# Using vLLM

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=bio_mistral_complete,
    llm_model_name=LLM_MODEL,
    llm_model_max_async=4,
    llm_model_kwargs={
        "base_url": "http://127.0.0.1:8080/v1", 
        "api_key": "none" 
    },
    chunk_token_size=1200,
    entity_extract_max_gleaning=0,
    default_embedding_timeout=120,
    
    embedding_func=EmbeddingFunc(
        embedding_dim=768, 
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            base_url="http://127.0.0.1:8081/v1",  # Updated port
            api_key="none",
            model=EMBEDDING_MODEL
        )
    )
)

# %%
async def main():
    await rag.initialize_storages()

    # %%
    # data_dir = os.path.join(project_root, 'biomed-rag', 'data', 'external', 'medqa')
    # Use path relative to the script for robustness
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data', 'external', 'medqa')
    sample_textbook = os.path.join(data_dir, 'textbooks', 'Anatomy_Gray.txt')

    with open(sample_textbook, 'r') as f:
        text = f.read()

    # %%
    await rag.ainsert(text)

if __name__ == "__main__":
    asyncio.run(main())


