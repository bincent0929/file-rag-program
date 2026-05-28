import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from rag import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT,
    clear,
    get_vectorstore,
    ingest,
)


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ingest_folder(folder_path: str):
    folder_path = folder_path.strip()
    if not folder_path:
        return "Enter a folder path."
    if not Path(folder_path).exists():
        return f"Folder not found: {folder_path}"
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            ingest(folder_path)
        return buf.getvalue() or "Ingestion complete."
    except Exception as e:
        return f"Error: {e}"


def clear_db():
    buf = io.StringIO()
    with redirect_stdout(buf):
        clear()
    return buf.getvalue() or "Done."


def chat(message: str, history):
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        yield "No documents ingested yet. Use the Ingest tab first."
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(message)
    context = _format_docs(docs)

    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    partial = ""
    for chunk in chain.stream({"context": context, "input": message}):
        partial += chunk
        yield partial

    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            page = doc.metadata.get("page", "")
            page_str = f" (page {page + 1})" if page != "" else ""
            sources.append(f"{source}{page_str}")

    if sources:
        yield partial + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)


with gr.Blocks(title="Local RAG") as demo:
    gr.Markdown("# Local RAG")

    with gr.Tabs():
        with gr.Tab("Ingest"):
            folder_input = gr.Textbox(
                label="Folder path",
                placeholder="/path/to/your/documents",
            )
            with gr.Row():
                ingest_btn = gr.Button("Ingest", variant="primary")
                clear_btn = gr.Button("Clear DB", variant="stop")
            status_box = gr.Textbox(label="Status", interactive=False, lines=10)

            ingest_btn.click(ingest_folder, inputs=folder_input, outputs=status_box)
            clear_btn.click(clear_db, inputs=None, outputs=status_box)

        with gr.Tab("Chat"):
            gr.ChatInterface(fn=chat)


if __name__ == "__main__":
    demo.launch()
