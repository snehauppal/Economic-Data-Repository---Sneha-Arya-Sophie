#!/usr/bin/env python3
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from pathlib import Path

# Model selection
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

print(f"Downloading or locating model {MODEL_REPO}...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="models")

# Model configuration
N_CTX, N_THREADS, N_BATCH = 4096, 4, 256
SYSTEM_PROMPT = "You are a helpful assistant. Provide clear, concise, and accurate answers."

print("Loading model into memory...")
llm = Llama(
    model_path=str(model_path),
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    verbose=False,
)

df = pd.read_csv("../data/etl_cleaned_dataset.csv")
documents = []
for _, row in df.iterrows():
    text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(text)



# Function to retrieve relevant documents (simple version)
def retrieve_context(question, k=5):
    hits = []
    q = question.lower()
    for doc in documents:
        if any(word in doc.lower() for word in q.split()):
            hits.append(doc)
        if len(hits) >= k:
            break
    return hits

# FastAPI application
app = FastAPI(title="Economic Data API", version="1.0")

class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
def ask(req: AskRequest):
# Retrieve relevant context (simple RAG)
    retrieved_docs = retrieve_context(req.question)

    context = "\n\n".join(retrieved_docs)

    # Build the prompt for Qwen (this mirrors your notebook logic)
    prompt = f"""
You are a helpful assistant.

Use ONLY the information in the context below to answer the question.
If the answer cannot be found in the context, say you do not know.

Context:
{context}

Question:
{req.question}

Answer:
"""

    # Query Qwen (same model, same setup)
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.0
    )

    answer = output["choices"][0]["message"]["content"]

    # Return answer + sources (relevant documents)
    return {
        "answer": answer,
        "sources": retrieved_docs[:3]  # show up to 3 sources
    }

@app.get("/")
def root():
    return {"status": "ok"}
