"""
Local RAG Pipeline
==================
Ingest documents → chunk → embed → store in Chroma → query with Ollama

Usage:
    python rag.py ingest ./docs        # ingest all files in a folder
    python rag.py query "your question" # ask a question
    python rag.py clear                 # wipe the vector DB and start fresh
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL   = "http://vrllm-server.taileedec.ts.net:11434"
EMBEDDING_MODEL   = "mxbai-embed-large"   # small, fast, runs locally via Ollama
LLM_MODEL         = "llama3"             # change to mistral, gemma2, etc.
CHROMA_DB_PATH    = "./chroma_db"        # where vectors are persisted on disk
CHUNK_SIZE        = 1000                 # characters per chunk
CHUNK_OVERLAP     = 150                  # overlap between chunks

# File extension → loader mapping
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".csv":  CSVLoader,
    ".md":   UnstructuredMarkdownLoader,
}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only \
the provided context from internal documents. If the answer is not in the context, \
say "I don't have that information in the documents provided."

Context:
{context}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
    )


def load_document(file_path: Path):
    """Pick the right loader based on file extension."""
    ext = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        print(f"  ⚠️  Skipping unsupported file type: {file_path.name}")
        return []
    print(f"  📄 Loading: {file_path.name}")
    loader = loader_cls(str(file_path))
    return loader.load()


# ── Commands ──────────────────────────────────────────────────────────────────

def ingest(folder: str):
    """Load all documents in a folder, chunk them, and store in Chroma."""
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    print(f"\n📂 Scanning: {folder_path.resolve()}\n")

    # 1. Load all documents
    all_docs = []
    for file_path in sorted(folder_path.rglob("*")):
        if file_path.is_file():
            docs = load_document(file_path)
            all_docs.extend(docs)

    if not all_docs:
        print("❌ No supported documents found.")
        sys.exit(1)

    print(f"\n✅ Loaded {len(all_docs)} document page(s)\n")

    # 2. Chunk
    print("✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries each in order
    )
    chunks = splitter.split_documents(all_docs)
    print(f"   → {len(chunks)} chunks created\n")

    # 3. Embed and store
    print(f"🔢 Embedding with '{EMBEDDING_MODEL}' and storing in Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"   → {vectorstore._collection.count()} total vectors in DB\n")
    print("✅ Ingestion complete!\n")


def query(question: str):
    """Query the vector store and generate an answer with the LLM."""
    vectorstore = get_vectorstore()

    if vectorstore._collection.count() == 0:
        print("❌ The vector DB is empty. Run `python rag.py ingest <folder>` first.")
        sys.exit(1)

    # Retriever — pulls top 4 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain — retrieve docs in parallel so we can return them as sources
    rag_chain = RunnableParallel(
        context=retriever,
        input=RunnablePassthrough(),
    ).assign(
        answer=(
            lambda x: {"context": format_docs(x["context"]), "input": x["input"]}
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n🔍 Question: {question}\n")
    result = rag_chain.invoke(question)

    print("💬 Answer:")
    print(result["answer"])

    print("\n📎 Sources:")
    seen = set()
    for doc in result["context"]:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            page = doc.metadata.get("page", "")
            page_str = f" (page {page + 1})" if page != "" else ""
            print(f"   • {source}{page_str}")
    print()


def clear():
    """Wipe the Chroma DB."""
    import shutil
    if Path(CHROMA_DB_PATH).exists():
        shutil.rmtree(CHROMA_DB_PATH)
        print("🗑️  Vector DB cleared.")
    else:
        print("ℹ️  Nothing to clear — DB doesn't exist yet.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python rag.py ingest <folder>")
            sys.exit(1)
        ingest(sys.argv[2])

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python rag.py query \"your question\"")
            sys.exit(1)
        query(" ".join(sys.argv[2:]))

    elif command == "clear":
        clear()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
