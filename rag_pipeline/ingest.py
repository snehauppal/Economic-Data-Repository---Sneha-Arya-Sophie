from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Embedding model loaded!")

# Detect the embedding dimension dynamically so FAISS always matches.
EMBED_DIM = embedder.get_sentence_embedding_dimension()
print("Embedding dimension:", EMBED_DIM)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen1.5-1.8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="auto"
)

