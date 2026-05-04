import os, json, requests, gc, time, sqlite3, faiss, numpy as np, networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ================= CONFIGURATION =================
BASE = r"E:\hybrid_rag"
FAISS = os.path.join(BASE, "data/processed/faiss_index.bin")
MAP = os.path.join(BASE, "data/processed/chunk_mapping.json")
CHUNKS = os.path.join(BASE, "data/processed/chunks.json")
DB = os.path.join(BASE, "data/processed/materials.db")
FACTS = os.path.join(BASE, "data/processed/extracted_facts.json")
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

# ================= CACHE LOADERS =================
@st.cache_resource
def load_system():
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    
    emb = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    idx = faiss.read_index(FAISS)
    with open(MAP, 'r') as f: mp = json.load(f)
    with open(CHUNKS, 'r', encoding='utf-8') as f: chunks = json.load(f)
    
    # Build lightweight KG from DB + extracted facts
    G = nx.DiGraph()
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT material, property_name, property_value, unit FROM material_data").fetchall()
    conn.close()
    
    for mat, prop, val, unit in rows:
        if not G.has_node(mat): G.add_node(mat, type="material")
        if not G.has_node(prop): G.add_node(prop, type="property")
        G.add_edge(mat, prop, relation="has_property", value=f"{val} {unit}")
        
    # Add literature facts if available
    try:
        with open(FACTS, 'r', encoding='utf-8') as f: facts = json.load(f)
        for fact in facts:
            m, p = fact.get("material"), fact.get("property")
            if m and p and G.has_node(m):
                G.add_edge(m, p, relation="literature_extract", value=fact.get("value"))
    except Exception:
        pass
        
    return emb, idx, mp, chunks, G

# ================= CORE ENGINE =================
def retrieve_semantic(q, emb, idx, mp, chunks, k=3):
    e = emb.encode([q]).astype('float32')
    d, ids = idx.search(e, k)
    res = []
    for i, id_ in enumerate(ids[0]):
        if id_ != -1 and id_ < len(mp):
            cid = mp[id_]
            c = next((x for x in chunks if x["chunk_id"] == cid), None)
            if c: res.append({"src": c["source_file"], "txt": c["text"][:400]+"...", "dist": float(d[0][i])})
    return res

def retrieve_structured(q, db_path, k=5):
    kw = [w.strip(".,;:!?\"'()[]{}") for w in q.split() if len(w) > 3]
    if not kw: return []
    conds, par = [], []
    for k in kw:
        conds.append("(material LIKE ? OR property_name LIKE ? OR source LIKE ?)")
        par.extend([f"%{k}%", f"%{k}%", f"%{k}%"])
    sql = f"SELECT * FROM material_data WHERE {' OR '.join(conds)} LIMIT {k}"
    try:
        conn = sqlite3.connect(db_path); cur = conn.cursor(); cur.execute(sql, par)
        rows, cols = cur.fetchall(), [d[0] for d in cur.description]; conn.close()
        return [{"data": dict(zip(cols, r))} for r in rows]
    except Exception: return []

def fuse(q, sem, sql):
    txt = [f"Query: {q}", "\n## 📚 Literature", ""]
    for i, r in enumerate(sem, 1): txt.extend([f"[{i}] {r['src']}", f"   {r['txt']}\n"])
    txt.extend(["## 🗄️ Structured Data", ""])
    for i, r in enumerate(sql, 1): txt.append(f"[{i}] {' | '.join(f'{k}: {v}' for k,v in r['data'].items())}")
    txt.extend([
        "\n## 📝 Instructions", "- Use ONLY provided evidence.", "- Cite (Lit)/(DB).",
        "- If agree → HIGH conf. If conflict → state range.", "- Focus on Al/SiC.",
        "- Keep <200 words."
    ])
    return "\n".join(txt)

def ask_ollama(ctx, temp=0.2):
    try:
        r = requests.post(OLLAMA, json={"model": MODEL, "messages": [{"role": "user", "content": ctx}],
                                        "stream": False, "options": {"temperature": temp, "num_ctx": 2048}}, timeout=90)
        r.raise_for_status(); return r.json()["message"]["content"]
    except Exception as e: return f"⚠️ Ollama Error: {e}"

def calibrate_confidence(sem_count, db_count, avg_sim=0.0, trend_r2=None):
    # Normalized similarity (FAISS distance is L2, lower=better. ~0-2 range typical)
    sim_norm = max(0, 1 - (avg_sim / 2.0))
    score = 0.0
    score += min(db_count * 12, 35)          # DB precision weight
    score += min(sem_count * 10, 30)         # Lit coverage weight
    score += sim_norm * 25                   # Relevance weight
    score += 10 if trend_r2 and trend_r2 > 0.8 else 0  # Trend fit bonus
    score = min(score, 100)
    label = "🟢 HIGH" if score >= 70 else ("🟡 MEDIUM" if score >= 40 else "🔴 LOW")
    breakdown = f"DB:{min(db_count*12,35)}% + Lit:{min(sem_count*10,30)}% + Sim:{sim_norm*25:.0f}%"
    return score, label, breakdown

# ================= UI =================
st.set_page_config(page_title="Hybrid RAG: Al/SiC Analysis", layout="wide")
st.title("🔬 Hybrid RAG: Al/SiC Composites")
st.caption("Local | 8GB RAM Optimized | Evidence-Backed | Calibrated Confidence")

if "history" not in st.session_state: st.session_state.history = []

tab_qa, tab_trend, tab_comp, tab_kg = st.tabs(["🔍 Q&A", "📈 Trends", "⚖️ Compare", "🔗 Knowledge Graph"])

# ---- TAB 1: Q&A ----
with tab_qa:
    q = st.text_input("Engineering Query:", placeholder="e.g., Effect of 10% SiC on tensile strength at 25°C")
    if st.button("Analyze", type="primary"):
        if q.strip():
            with st.spinner("Retrieving & synthesizing..."):
                t0 = time.time()
                emb, idx, mp, chunks, G = load_system()
                sem = retrieve_semantic(q, emb, idx, mp, chunks, k=3)
                sql = retrieve_structured(q, DB, k=5)
                ctx = fuse(q, sem, sql)
                ans = ask_ollama(ctx)
                
                avg_dist = np.mean([r["dist"] for r in sem]) if sem else 2.0
                score, label, breakdown = calibrate_confidence(len(sem), len(sql), avg_dist)
                elapsed = f"{time.time()-t0:.1f}s"
                
            st.markdown(ans)
            st.info(f"🎯 Confidence: **{label}** ({score:.0f}/100) | {breakdown} | ⏱️ {elapsed}")
            
            st.session_state.history.insert(0, {"q": q, "ans": ans, "score": score, "ts": datetime.now().strftime("%H:%M")})
            if len(st.session_state.history) > 15: st.session_state.history.pop()

    # Export Button
    if st.session_state.history:
        latest = st.session_state.history[0]
        ev_txt = f"Lit chunks: {len(sem)} | DB matches: {len(sql)}\n" + "\n".join([f"- {r}" for r in sem[:2]]) if sem else "No lit matches"
        report_md = f"# Analysis Report\n**Query:** {latest['q']}\n**Confidence:** {latest['score']:.0f}%\n**Response:**\n{latest['ans']}\n\n## Evidence\n{ev_txt}"
        st.download_button("📥 Export Report (.md)", report_md.encode('utf-8'), 
                           file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown")

# ---- TAB 2: TRENDS ----
with tab_trend:
    col1, col2 = st.columns(2)
    base = col1.text_input("Base Material", "Al_A356")
    param = col2.text_input("Varying Keyword", "wt_SiC")
    prop = st.selectbox("Property", ["tensile_strength", "yield_strength", "hardness", "elongation"])
    if st.button("Generate Trend", type="primary"):
        # Inline trend logic (lightweight, no extra files)
        conn = sqlite3.connect(DB)
        rows = conn.execute("SELECT material, property_value FROM material_data WHERE material LIKE ? AND property_name = ?", [f"%{base}%", prop]).fetchall()
        conn.close()
        if len(rows) >= 3:
            import re
            xs, ys, labs = [], [], []
            for m, v in rows:
                match = re.search(rf'(\d+(?:\.\d+)?)\s*{param}', m, re.I)
                if match: xs.append(float(match.group(1))); ys.append(float(v)); labs.append(m)
            if len(xs) >= 3:
                p1 = np.polyfit(xs, ys, 1); y1 = np.polyval(p1, xs); r2 = 1 - np.sum((ys-y1)**2)/np.sum((ys-np.mean(ys))**2)
                slope = p1[0]
                st.success(f"📈 {prop} {'increases' if slope>0 else 'decreases'} with {param} (R² = {r2:.2f})")
                fig, ax = plt.subplots(figsize=(5,3)); ax.scatter(xs,ys,c="crimson"); ax.plot(xs,y1,"b--"); ax.set_xlabel(param); ax.set_ylabel(prop); ax.grid(True, alpha=0.3)
                st.pyplot(fig); plt.close('all')
            else: st.warning("Could not extract numeric values from material names.")
        else: st.warning(f"Need ≥3 data points. Found: {len(rows)}")

# ---- TAB 3: COMPARISON ----
with tab_comp:
    mats = st.text_area("Materials (one per line)", "Al_A356_Pure\nAl_A356_10wt_SiC\nAl_A356_20wt_SiC")
    props = st.text_input("Properties (comma separated)", "tensile_strength, hardness")
    if st.button("Compare", type="primary"):
        mat_list = [m.strip() for m in mats.split("\n") if m.strip()]
        prop_list = [p.strip() for p in props.split(",") if p.strip()]
        conn = sqlite3.connect(DB)
        res_data = {}
        for m in mat_list:
            res_data[m] = {}
            for p in prop_list:
                row = conn.execute("SELECT property_value, unit FROM material_data WHERE material = ? AND property_name = ?", (m,p)).fetchone()
                res_data[m][p] = row if row else None
        conn.close()
        st.json(res_data)
        if len(mat_list) >= 2 and all(res_data.get(mat_list[0],{}).get(p) for p in prop_list if res_data.get(mat_list[0],{}).get(p)):
            base = mat_list[0]
            for m in mat_list[1:]:
                for p in prop_list:
                    b_val, c_val = res_data[base][p][0] if res_data[base].get(p) else 0, res_data[m][p][0] if res_data[m].get(p) else 0
                    if b_val and b_val != 0:
                        pct = ((c_val - b_val)/b_val)*100
                        st.markdown(f"🔹 {m}: **{pct:+.1f}%** {p} vs {base}")

# ---- TAB 4: KNOWLEDGE GRAPH ----
with tab_kg:
    st.subheader("🔗 Material-Property Relationship Explorer")
    _, idx, _, _, G = load_system()
    start_node = st.text_input("Start from (e.g., Al_A356_Pure or tensile_strength)")
    if st.button("Find Connections"):
        if G.has_node(start_node):
            neighbors = list(G.successors(start_node))
            edges_info = [f"→ {n} ({G.edges[start_node,n]['relation']}, {G.edges[start_node,n]['value']})" for n in neighbors]
            if edges_info:
                st.success(f"Found {len(neighbors)} connections:")
                for e in edges_info: st.markdown(f"- {e}")
            else: st.info("No outgoing edges found.")
        else:
            st.warning(f"Node '{start_node}' not in graph. Try: {list(G.nodes())[:5]}")
    gc.collect()

gc.collect()