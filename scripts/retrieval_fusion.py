import os
import json
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = r"E:\hybrid_rag"
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "processed", "faiss_index.bin")
CHUNK_MAPPING_PATH = os.path.join(BASE_DIR, "data", "processed", "chunk_mapping.json")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "materials.db")

class HybridRetrievalFusion:
    def __init__(self):
        # Load FAISS index & mapping
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNK_MAPPING_PATH, 'r') as f:
            self.chunk_mapping = json.load(f)
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load embedding model (kept in memory for fast querying)
        os.environ["OMP_NUM_THREADS"] = "2"
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.embedder.max_seq_length = 512

    def retrieve_semantic(self, query, k=3):
        """Search FAISS for semantically similar chunks"""
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.faiss_index.search(q_emb, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.chunk_mapping):
                chunk_id = self.chunk_mapping[idx]
                chunk_data = next((c for c in self.chunks if c["chunk_id"] == chunk_id), None)
                if chunk_data:
                    results.append({
                        "type": "literature",
                        "source": chunk_data["source_file"],
                        "chunk_id": chunk_id,
                        "text": chunk_data["text"][:500] + "...",  # Truncate for safety
                        "score": float(distances[0][indices[0].tolist().index(idx)])
                    })
        return results

    def retrieve_structured(self, query, k=5):
        """Search SQLite for exact matches"""
        # Simple keyword extraction: split query, filter out short/common words
        keywords = [w.strip(".,;:!?\"'") for w in query.split() if len(w) > 3]
        
        # Build safe parameterized query
        conditions = []
        params = []
        for kw in keywords:
            conditions.append("(material LIKE ? OR property_name LIKE ? OR source LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])
        
        if not conditions:
            return []
            
        sql = f"SELECT * FROM material_data WHERE {' OR '.join(conditions)} LIMIT {k}"
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            conn.close()
            
            results = []
            for row in rows:
                results.append({"type": "database", "data": dict(zip(col_names, row))})
            return results
        except Exception as e:
            print(f"⚠️ SQL retrieval error: {e}")
            return []

    def fuse_results(self, query, sem_results, sql_results):
        """Combine, deduplicate, tag, and compress for LLM"""
        output_lines = [f"Query: {query}", "\n## 📚 Literature Evidence (Semantic)", ""]
        
        if sem_results:
            for i, res in enumerate(sem_results, 1):
                output_lines.append(f"[{i}] {res['source']}")
                output_lines.append(f"   Content: {res['text']}")
                output_lines.append("")
        else:
            output_lines.append("   No highly relevant literature chunks found.")
            output_lines.append("")
            
        output_lines.append("## 🗄️ Structured Data (Exact)")
        output_lines.append("")
        
        if sql_results:
            for i, res in enumerate(sql_results, 1):
                data = res["data"]
                line_parts = [f"{k}: {v}" for k, v in data.items()]
                output_lines.append(f"[{i}] {' | '.join(line_parts)}")
        else:
            output_lines.append("   No exact database matches for query keywords.")
            
        output_lines.append("\n## 📝 Instructions for AI")
        output_lines.append("- Synthesize findings using ONLY the provided evidence.")
        output_lines.append("- If literature and DB agree, state confidence as HIGH.")
        output_lines.append("- If they differ, note the discrepancy and cite sources.")
        output_lines.append("- Focus on Al/SiC composite behavior (strength, temp, processing, microstructure).")
        output_lines.append("- Keep response < 250 words. Use bullet points if helpful.")
        
        return "\n".join(output_lines)

# Quick test function
if __name__ == "__main__":
    print("🚀 Initializing Hybrid Retrieval & Fusion...")
    system = HybridRetrievalFusion()
    
    test_query = "How does SiC reinforcement volume fraction affect tensile strength of Aluminum composites at high temperatures?"
    print(f"\n🔍 Query: {test_query}")
    
    sem = system.retrieve_semantic(test_query, k=3)
    sql = system.retrieve_structured(test_query, k=5)
    
    print(f"✅ Found {len(sem)} literature chunks, {len(sql)} DB records")
    
    fused = system.fuse_results(test_query, sem, sql)
    print("\n" + "="*60)
    print("FUSED CONTEXT (Ready for LLM):")
    print("="*60)
    print(fused)