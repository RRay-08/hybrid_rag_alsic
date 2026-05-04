import os
import json
import gc
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = r"E:\hybrid_rag"
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "processed", "faiss_index.bin")
CHUNK_MAPPING_PATH = os.path.join(BASE_DIR, "data", "processed", "chunk_mapping.json")
BATCH_SIZE = 32  # Small batches to prevent RAM spikes

def build_faiss_index():
    print("📖 Loading chunks...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"✅ Loaded {len(chunk_texts)} chunks")

    # Load embedding model on CPU with limited threads
    print("🧠 Loading embedding model (this takes ~30s)...")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    model.max_seq_length = 512

    # Encode in batches
    print("⚙️ Generating embeddings in batches...")
    all_embeddings = []
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i:i+BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(emb)
        gc.collect()  # Clean up after each batch
    embeddings = np.vstack(all_embeddings).astype('float32')

    # Clear model from memory immediately
    print("🧹 Freeing embedding model memory...")
    del model
    torch.cuda.empty_cache()  # Safe to call even on CPU
    gc.collect()

    # Build FAISS index
    print("🗃️ Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save chunk mapping (index position → chunk_id)
    with open(CHUNK_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunk_ids, f)

    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"📊 Mapping saved: {len(chunk_ids)} chunks indexed")
    print("✅ build_vector_index.py completed successfully")

if __name__ == "__main__":
    build_faiss_index()