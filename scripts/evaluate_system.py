import sys, os, json, time, sqlite3, faiss, gc
import requests
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, sys_path)

BASE = r"E:\hybrid_rag"
FAISS = os.path.join(BASE, "data", "processed", "faiss_index.bin")
MAP = os.path.join(BASE, "data", "processed", "chunk_mapping.json")
CHUNKS = os.path.join(BASE, "data", "processed", "chunks.json")
DB = os.path.join(BASE, "data", "processed", "materials.db")
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

print("🔧 Loading components...")
emb = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
faiss_idx = faiss.read_index(FAISS)
with open(MAP, 'r') as f: mp = json.load(f)
with open(CHUNKS, 'r', encoding='utf-8') as f: chunks = json.load(f)
print("✅ Components loaded")

# ================= OLLAMA HEALTH CHECK =================
def test_ollama_connection():
    """Verify Ollama is running and model is available"""
    try:
        # Ping Ollama
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        if r.status_code != 200:
            print(f"❌ Ollama API returned status {r.status_code}")
            return False
        
        # Check if our model exists
        models = r.json().get("models", [])
        model_names = [m["name"] for m in models]
        if MODEL not in model_names:
            print(f"❌ Model '{MODEL}' not found. Available: {model_names}")
            print(f"💡 Fix: Run 'ollama pull {MODEL}' in PowerShell")
            return False
        print(f"✅ Ollama connected | Model '{MODEL}' ready")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("💡 Fix: Open Start Menu → search 'Ollama' → run it")
        return False
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False

# ================= RETRIEVAL FUNCTIONS =================
def retrieve_semantic(q, k=2):
    e = emb.encode([q]).astype('float32')
    _, ids = faiss_idx.search(e, k)
    txt = []
    for i in ids[0]:
        if i != -1 and i < len(mp):
            cid = mp[i]
            c = next((chunk for chunk in chunks if chunk["chunk_id"] == cid), None)
            if c:
                txt.append(c["text"][:400])  # Truncate for debug display
    return txt

def retrieve_structured(q, k=3):
    kw = [w.strip(".,;:!?\"'()[]{}") for w in q.split() if len(w) > 3]
    if not kw:
        return []
    conds, par = [], []
    for word in kw:
        conds.append("(material LIKE ? OR property_name LIKE ? OR source LIKE ?)")
        par.extend([f"%{word}%", f"%{word}%", f"%{word}%"])
    sql = f"SELECT * FROM material_data WHERE {' OR '.join(conds)} LIMIT {k}"
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute(sql, par)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        conn.close()
        return [" | ".join(f"{c}: {v}" for c, v in zip(cols, row)) for row in rows]
    except Exception as e:
        print(f"⚠️ SQL error: {e}")
        return []

# ================= SCORING =================
def score_response(resp, keywords):
    if not resp or resp.strip() == "":
        return 0.0
    low = resp.lower()
    hits = sum(1 for k in keywords if k.lower() in low)
    return min(hits / max(len(keywords), 1), 1.0)

# ================= MAIN EVALUATION =================
def run_evaluation():
    # Ground truth tests (ADJUST KEYWORDS TO MATCH YOUR ACTUAL DATA)
    TESTS = [
    	{"q": "What is the tensile strength of Al A356 with 10% SiC at room temperature?", "keywords": ["210", "MPa", "SiC"]},
    	{"q": "How does adding SiC affect tensile strength of Al A356?", "keywords": ["increase", "strength", "SiC"]},
    	{"q": "What is the optimal temperature for heat treatment of Al matrix composites?", "keywords": ["heat", "treatment","temperature"]},
   	{"q": "Does SiC reinforcement improve hardness of aluminum composites?", "keywords": ["hardness", "improve", "SiC"]},
    	{"q": "Which property decreases at high temperatures in Al/SiC composites?", "keywords": ["decrease", "strength", "temperature"]}
	]

    print("\n🧪 Running Evaluation...\n")
    scores = []
    
    for i, t in enumerate(TESTS, 1):
        print(f"[{i}/5] Query: {t['q']}")
        
        # Retrieve evidence
        sem_chunks = retrieve_semantic(t["q"], k=2)
        sql_rows = retrieve_structured(t["q"], k=3)
        
        print(f"   📚 Retrieved: {len(sem_chunks)} lit chunks, {len(sql_rows)} DB rows")
        if sem_chunks:
            print(f"   Sample lit: '{sem_chunks[0][:100]}...'")
        if sql_rows:
            print(f"   Sample DB: '{sql_rows[0][:100]}...'")
        
        # Build context
        ctx = f"Evidence:\n{' '.join(sem_chunks)}\n{' '.join(sql_rows)}\n\nQuery: {t['q']}\nAnswer in one short sentence using ONLY the evidence above:"
        
        # Call Ollama with detailed error handling
        try:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": ctx}],
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 2048}
            }
            print(f"   🤖 Sending to Ollama (timeout=120s)...")
            start = time.time()
            r = requests.post(OLLAMA, json=payload, timeout=120)
            elapsed = time.time() - start
            
            if r.status_code != 200:
                print(f"   ❌ HTTP {r.status_code}: {r.text[:200]}")
                ans = "ERROR"
            else:
                resp_json = r.json()
                if "message" not in resp_json or "content" not in resp_json["message"]:
                    print(f"   ❌ Unexpected response format: {resp_json}")
                    ans = "ERROR"
                else:
                    ans = resp_json["message"]["content"].strip()
                    print(f"   ✅ Response ({elapsed:.1f}s): {ans[:120]}...")
                    
        except requests.exceptions.Timeout:
            print(f"   ❌ TIMEOUT: Ollama took >120s. Try reducing context or upgrading RAM.")
            ans = "TIMEOUT"
        except requests.exceptions.ConnectionError:
            print(f"   ❌ CONNECTION ERROR: Ollama not reachable. Is it running?")
            ans = "CONNECTION_ERROR"
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON ERROR: {e} | Raw response: {r.text[:200]}")
            ans = "JSON_ERROR"
        except Exception as e:
            print(f"   ❌ UNEXPECTED: {type(e).__name__}: {e}")
            ans = "UNKNOWN_ERROR"
        
        # Score
        s = score_response(ans, t["keywords"])
        scores.append(s)
        print(f"   📊 Match: {s:.0%} | Keywords found: {sum(1 for k in t['keywords'] if k.lower() in ans.lower())}/{len(t['keywords'])}\n")
        gc.collect()
    
    # Summary
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n📊 FINAL: Average Accuracy = {avg:.1%}")
    print(f"💡 If accuracy is low:")
    print(f"   1. Check if keywords match your ACTUAL data (CSV/papers)")
    print(f"   2. Try simpler test queries first")
    print(f"   3. Increase timeout if CPU is slow")
    print(f"   4. Verify Ollama model is quantized (q4_0) for speed")

if __name__ == "__main__":
    if test_ollama_connection():
        run_evaluation()
    else:
        print("\n🛑 Evaluation aborted. Fix Ollama connection first.")