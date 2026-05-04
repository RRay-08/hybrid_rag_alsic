@echo off

echo 🚀 Starting Hybrid RAG System...

echo 📁 Directory: E:\hybrid_rag

cd /d E:\hybrid_rag

call venv\Scripts\activate.bat

echo ✅ Environment activated

echo 🌐 Opening browser...

start http://localhost:8501

streamlit run app.py

pause