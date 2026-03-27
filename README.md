# 🧠 Codebase Onboarding Assistant

A RAG-powered tool that ingests any GitHub repository and answers natural-language questions about the codebase using a LangChain agent with GPT-4o.

## ✨ Features

- **Multi-language code ingestion** — Python, JavaScript, TypeScript, Java, Go, and more with code-aware chunking
- **Semantic code search** — find relevant code snippets by meaning, not just keywords
- **File explanation** — get structured explanations of any file in the repo
- **Onboarding wiki generation** — auto-generate a markdown onboarding guide for new developers
- **Tool-use agent** — GPT-4o decides which tool to call based on your query
- **Citation-backed answers** — every response cites exact file paths

## 🏗️ Architecture

```
codebase-onboarding-assistant/
├── app.py                  # Streamlit frontend
├── rag/
│   ├── ingestor.py         # Clone repo → chunk → embed → Chroma
│   ├── retriever.py        # Query Chroma, return relevant chunks
│   └── chain.py            # LangChain agent with tools
├── tools/
│   ├── search_code.py      # Tool: semantic search over codebase
│   ├── explain_file.py     # Tool: explain a specific file
│   └── generate_wiki.py    # Tool: generate onboarding wiki
├── chroma_db/              # Persisted vector store (gitignored)
└── .env                    # API keys (gitignored)
```

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/your-username/codebase-onboarding-assistant.git
cd codebase-onboarding-assistant
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   GITHUB_TOKEN=ghp_...     (optional, for private repos)
```

### 3. Run
```bash
streamlit run app.py
```

### 4. Use
1. Paste a GitHub repo URL in the sidebar and click **Ingest Repo**
2. Ask questions: *"How does authentication work?"*, *"What does the PaymentService class do?"*
3. Click **Generate Onboarding Wiki** to get a downloadable markdown guide

## 🔧 Tech Stack

| Layer | Tool |
|---|---|
| LLM | GPT-4o |
| Embeddings | text-embedding-3-small |
| Vector DB | Chroma (persistent) |
| Framework | LangChain |
| Code ingestion | GitHub API + GitPython |
| Code chunking | LangChain `Language` splitter |
| Frontend | Streamlit |

## 📄 License

MIT
