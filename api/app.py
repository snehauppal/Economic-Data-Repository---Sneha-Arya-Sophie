#!/usr/bin/env python3
#imports
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from pathlib import Path

# # Hugging Face repository and model file for the local Qwen model
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

print(f"Downloading or locating model {MODEL_REPO}...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="models")

# Model configuration
N_CTX, N_THREADS, N_BATCH = 4096, 4, 256
SYSTEM_PROMPT = "You are a helpful assistant. Provide clear, concise, and accurate answers."

#Load the Qwen model into memory

print("Loading model into memory...")
llm = Llama(
    model_path=str(model_path),
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    verbose=False,
)
#Loading the csv data directly - we loaded a cleaned CSV produced by our ETL pipeline
df = pd.read_csv("../data/etl_cleaned_dataset.csv")
# Convert each row of the dataset into a text document
documents = []
for _, row in df.iterrows():
    text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(text)



# Function to retrieve relevant documents
def retrieve_context(question, k=5):
    hits = []
    q = question.lower()
    # Check each document for overlap with the query terms
    for doc in documents:
        if any(word in doc.lower() for word in q.split()):
            hits.append(doc)
        # Stop once we have enough context documents
        if len(hits) >= k:
            break
    return hits

# FastAPI application
app = FastAPI(title="Economic Data API", version="1.0")
# Enforces that every request includes a "question" field
class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
def ask(req: AskRequest):
# Retrieve relevant context
    retrieved_docs = retrieve_context(req.question)

    context = "\n\n".join(retrieved_docs)

    # Builds the prompt
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

    # Query Qwen
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.0
    )
# Extract the model's response text
    answer = output["choices"][0]["message"]["content"]

    # Returns the answer and a small set of source documents
    return {
        "answer": answer,
        "sources": retrieved_docs[:3]  # show up to 3 sources
    }

@app.get("/")
def root():
    return {"status": "ok"}


#CURL Testing example and server command
# The following curl command demonstrates how to query the API
# once the server is running with this server command
#   uvicorn app:app --host 0.0.0.0 --port 8000
#
# curl -X POST http://localhost:8000/api/ask \
#   -H "Content-Type: application/json" \
#   -d '{
#         "question": "What was the unemployment rate in the United States in 2019?"
#       }'
#
# Expected response:
# {
#   "answer": "The unemployment rate in the United States in 2019 was 3.7%.",
#   "sources": [...]
# }
