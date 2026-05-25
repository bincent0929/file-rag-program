# Local RAG Pipeline

A fully local, private RAG system. Your files never leave your machine.

```
Your files → LangChain (load + chunk) → nomic-embed-text (vectorize) → Chroma (store) → Llama3 (answer)
```

---

## Prerequisites

### 1. Install Ollama
Download from https://ollama.com and install it.

### 2. Pull the required models
```bash
ollama pull llama3              # the LLM that answers questions
ollama pull nomic-embed-text    # the embedding model that vectorizes text
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Ingest documents
Point it at a folder — it will recursively find and ingest all supported files.
```bash
python rag.py ingest ./docs
```

Supported file types: `.pdf`, `.docx`, `.txt`, `.csv`, `.md`

### Ask a question
```bash
python rag.py query "What is our refund policy?"
python rag.py query "Summarize the Q3 report"
python rag.py query "Who is the contact for HR issues?"
```

### Clear the database
Wipes all stored vectors so you can start fresh.
```bash
python rag.py clear
```

---

## How it works

1. **Load** — LangChain reads each file using a format-specific loader (PDF, DOCX, etc.)
2. **Chunk** — `RecursiveCharacterTextSplitter` splits documents into overlapping 1000-char chunks
3. **Embed** — `nomic-embed-text` (running locally via Ollama) converts each chunk into a vector
4. **Store** — Chroma saves vectors to disk in `./chroma_db/`
5. **Query** — Your question is embedded, the 4 closest chunks are retrieved, and Llama3 generates an answer grounded in those chunks

---

## Configuration

Edit the constants at the top of `rag.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3` | Ollama model for answering (try `mistral`, `gemma2`) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for vectorizing |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `CHROMA_DB_PATH` | `./chroma_db` | Where vectors are stored on disk |

---

## Tips

- **More accurate answers** → increase `k` in `search_kwargs={"k": 4}` to retrieve more chunks
- **Faster responses** → swap `llama3` for `mistral` (smaller, quicker)
- **Re-ingesting** → run `python rag.py clear` first to avoid duplicate vectors
- **Large document sets** → chunking and embedding only happens once; querying is always fast
