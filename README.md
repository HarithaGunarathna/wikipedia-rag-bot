# Wikipedia RAG Bot

A fully local RAG (Retrieval-Augmented Generation) chatbot that answers questions from Wikipedia. Repeated or similar questions are served from a persistent ChromaDB cache — no re-fetching needed.

## Architecture

```
User question
     │
     ▼
Semantic search in ChromaDB (persistent cache)
     │
     ├─── Cache hit (similarity ≥ 0.45) ──────────────────────┐
     │                                                          │
     └─── Cache miss                                           │
               │                                               │
               ▼                                               │
     Wikipedia search (top N articles)                         │
               │                                               │
               ▼                                               │
     Text chunking (500 tokens, 50 overlap)                    │
               │                                               │
               ▼                                               │
     Embed chunks (all-MiniLM-L6-v2, local)                   │
               │                                               │
               ▼                                               │
     Store in ChromaDB (deduplicated by content hash)          │
               │                                               │
               └───────────────────────────────────────────────┘
                                    │
                                    ▼
                    Retrieve top-4 relevant chunks
                                    │
                                    ▼
                    Ollama LLM (Llama 3 / Mistral / etc.)
                                    │
                                    ▼
                    Answer + Wikipedia source links
```

## Tech stack

| Component | Library |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store / cache | ChromaDB (persistent, local) |
| LLM | Ollama (local) |
| RAG orchestration | LangChain (LCEL) |
| Wikipedia | `wikipedia` |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

Download from [ollama.com](https://ollama.com), then:

```bash
ollama serve
ollama pull llama3   # or mistral, phi3, gemma3:1b, etc.
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded automatically on first launch.

## Usage

1. Type any question in the chat box
2. The bot checks ChromaDB first — if a semantically similar topic is cached, it answers instantly (🟢 **From cache**)
3. On a cache miss, it fetches Wikipedia, stores the embeddings, and answers (🌐 **From Wikipedia**)
4. Expand **Wikipedia sources** under any answer to see which articles were used

## Sidebar settings

| Setting | Description |
|---|---|
| Ollama model | Which local model to use (must be pulled first) |
| Wikipedia pages to fetch | 1–5 articles per cache miss; more = richer context but slower |
| Cached chunks | Number of text chunks currently stored in ChromaDB |
| Clear cache | Wipes the ChromaDB collection so all questions re-fetch from Wikipedia |

## Project structure

```
wikipedia-rag-bot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG logic (cache, Wikipedia fetch, LLM chain)
├── requirements.txt
├── chroma_db/          # Persistent ChromaDB store (auto-created)
└── .streamlit/
    └── config.toml
```
