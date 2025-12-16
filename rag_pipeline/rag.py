import faiss
import numpy as np

# Create a simple L2 index with the correct dimension
index = faiss.IndexFlatL2(EMBED_DIM)

# Store raw texts and simple metadata
documents = []   # list of text
sources = []     # where each text came from (manual, pdf filename, spreadsheet name, etc.)

print("FAISS index created with dimension:", EMBED_DIM)
print("Qwen model loaded!")

from typing import List
from pypdf import PdfReader
import pandas as pd
from google.colab import files


def add_document(text: str, source: str = "manual"):
    """Embed the text, add it to FAISS, and record its source.

    text   : the raw text to store
    source : a short label like 'manual', 'myfile.pdf', or 'grades.xlsx'
    """
    if not text.strip():
        print("Skipped empty text.")
        return

    embedding = embedder.encode([text])[0].astype("float32")
    index.add(np.array([embedding]))
    documents.append(text)
    sources.append(source)
    print(f"Added document #{len(documents)} from source: {source} (length={len(text)} chars)")


def upload_pdf():
    """Upload one or more PDFs, extract all text, and add each as a document.

    Each uploaded PDF becomes one document in FAISS.
    For more advanced use, you could split large PDFs into chunks.
    """
    uploaded = files.upload()
    for filename in uploaded.keys():
        reader = PdfReader(filename)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"
        print(f"[PDF] Extracted {len(full_text)} characters from {filename}")
        add_document(full_text, source=filename)


def upload_spreadsheet():
    """Upload one or more spreadsheets (CSV or Excel) and add them as text documents.

    Strategy:
    - If CSV: read with pandas.read_csv.
    - If Excel: read with pandas.read_excel (requires openpyxl).
    - Convert DataFrame to CSV-style text string.
    - Store that text in FAISS so it can be retrieved semantically.
    """
    uploaded = files.upload()
    for filename in uploaded.keys():
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(filename)
        else:
            df = pd.read_excel(filename)
        text = df.to_csv(index=False)
        print(f"[Spreadsheet] {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        add_document(text, source=filename)


print("Helper functions defined: add_document(), upload_pdf(), upload_spreadsheet()")

def rag_query(query: str, top_k: int = 3, show_sources: bool = True) -> str:
    # 1. Embed the query
    q_embedding = embedder.encode([query])[0].astype("float32")

    # 2. Search FAISS
    distances, indices = index.search(np.array([q_embedding]), top_k)

    # 3. Build context from the closest documents
    retrieved_docs = []
    retrieved_meta = []
    for idx in indices[0]:
        if 0 <= idx < len(documents):
            retrieved_docs.append(documents[idx])
            retrieved_meta.append(sources[idx])

    context = "\n\n---\n\n".join(retrieved_docs)

    if show_sources:
        print("Retrieved from sources:")
        for m in retrieved_meta:
            print(" -", m)
        print()

    # 4. Build prompt
    prompt = f"""You are a helpful assistant.

You will be given some context from documents, followed by a question.
Use only the information in the context to answer as accurately and precisely as possible. Pay close attention to numerical values and reproduce them exactly as found in the context.

Context:
{context}

Question: {query}

Answer:
"""

    # 5. Generate answer with Qwen
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.0 # Set temperature to 0.0 for deterministic output
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

upload_spreadsheet()
