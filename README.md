# hybrid_rag_alsic
Hybrid RAG system for Al/SiC composite analysis - Local deployment, 8GB RAM optimized
# 🔬 Hybrid RAG System for Al/SiC Composites

A locally-hosted, evidence-backed AI assistant for materials engineering that combines semantic search over scientific literature with exact querying of structured property databases.

## ✨ Features

- **Hybrid Retrieval**: FAISS (semantic) + SQLite (structured) for comprehensive answers
- **8GB RAM Optimized**: Runs entirely on consumer hardware
- **Evidence-Backed**: Every claim cites literature or database sources
- **Calibrated Confidence**: Quantitative trust scores (0-100) with explainable breakdowns
- **Trend Analysis**: Automatic property vs. parameter fitting with R² metrics
- **Knowledge Graph**: Multi-hop reasoning over material-property-process relationships
- **Report Export**: One-click Markdown reports with full evidence trails

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama (https://ollama.com)
- 8GB RAM minimum

### Installation

1. Clone repository:
```bash
git clone https://github.com/YOUR_USERNAME/hybrid_rag_alsic.git
cd hybrid_rag_alsic
