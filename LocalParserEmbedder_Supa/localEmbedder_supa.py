
import os
from pathlib import Path
import hashlib
from langchain.embeddings import HuggingFaceBgeEmbeddings
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np
import torch

# Load environment variables
SUPABASE_URL = "https://rrzxyhgddphzsnnsfhjx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJyenh5aGdkZHBoenNubnNmaGp4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3MTM2ODgsImV4cCI6MjA2NTI4OTY4OH0.S-VOCbAzsOyOUlIzi6sjypOt2B-E57Lf3Huau6LJKc0"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Detect Apple Silicon and set device accordingly
if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# Get embedding dim
# Use embed_query for a single string
embedding_vec = model.embed_query("test")
dim = len(embedding_vec)
print(f"Embedding dimension: {dim}")

# 3. Chunking utility
def chunk_text(text, max_chars=2000):
    return [text[i: i+max_chars] for i in range(0, len(text), max_chars)]

# Helper function to generate valid integer point IDs
def generate_point_id(doc_name, chunk_idx):
    unique_string = f"{doc_name}_{chunk_idx}"
    hash_object = hashlib.md5(unique_string.encode())
    return int.from_bytes(hash_object.digest()[:8], byteorder='big')

# 4. Process PDFs from folder
folder_path = Path("./pdfs")  
pdf_files = list(folder_path.glob("*.pdf"))

rows = []
for pdf_path in tqdm(pdf_files):
    try:
        elements = partition_pdf(filename=str(pdf_path))
        text = "\n".join([e.text for e in elements if hasattr(e, "text") and e.text])
        chunks = chunk_text(text)
        embeddings = model.embed_documents(chunks)

        for idx, vec in enumerate(embeddings):
            row = {
                "content": chunks[idx],
                "metadata": {"file_path": str(pdf_path), "chunk_idx": idx},
                "embedding": vec,
            }
            rows.append(row)
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")

# Insert in batches
for i in range(0, len(rows), 500):
    batch = rows[i:i+500]
    # Supabase expects a list of dicts, embedding as list
    data = supabase.table("documents").insert(batch).execute()

print(f"Inserted {len(rows)} vector chunks from {len(pdf_files)} PDFs.")
