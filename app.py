import os
import gradio as gr
import chromadb
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.ocr_utils import run_ocr
from app.services.ML_Model import predict_text

# ---------------------------------------------------------------------------
# HuggingFace Inference Client  (add HF_TOKEN as a Space secret)
# ---------------------------------------------------------------------------
_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY", "")
_LLM_MODEL = "deepseek-ai/DeepSeek-R1:novita"

_hf_client = InferenceClient(api_key=_HF_TOKEN) if _HF_TOKEN else None


def _call_llm(prompt: str) -> str:
    """Send a prompt to the HF inference API and return the response text."""
    if _hf_client is None:
        return (
            "HF_TOKEN is not set. Please add it as a Secret in your Space settings "
            "(Settings -> Variables and secrets -> New secret: HF_TOKEN)."
        )
    try:
        completion = _hf_client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as exc:
        return f"LLM error: {exc}"


# ---------------------------------------------------------------------------
# Per-request RAG helper - builds an ephemeral ChromaDB from one PDF file
# ---------------------------------------------------------------------------
def _build_collection_from_pdf(file_path: str):
    """Load a PDF, chunk it, and return an ephemeral ChromaDB collection."""
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(raw_docs)

    # EphemeralClient: in-memory, isolated per request, HF Spaces safe
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="uploaded_pdf")

    documents = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    batch = 50
    for i in range(0, len(documents), batch):
        collection.upsert(
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
        )

    return collection


# ---------------------------------------------------------------------------
# Tab handler functions
# ---------------------------------------------------------------------------

def ocr_fn(file_path: str, lang: str) -> str:
    if file_path is None:
        return "Please upload a file."
    try:
        return run_ocr(file_path, lang.strip() or "eng")
    except Exception as exc:
        return f"Error: {exc}"


def rag_fn(file_path, question: str) -> str:
    if file_path is None:
        return "Please upload a PDF."
    if not question or not question.strip():
        return "Please enter a question."
    try:
        collection = _build_collection_from_pdf(file_path)
        results = collection.query(query_texts=[question], n_results=4)
        context_docs = results["documents"][0]

        prompt = (
            "You are a helpful assistant. Answer ONLY using the provided context. "
            "If the answer is not in the context, say \"I don't know.\"\n\n"
            "Context:\n" + "\n".join(context_docs) + "\n\n"
            "Question: " + question
        )
        return _call_llm(prompt)
    except Exception as exc:
        return f"Error: {exc}"


def extract_fn(file_path, query: str, fields: str) -> str:
    if file_path is None:
        return "Please upload a PDF."
    if not fields or not fields.strip():
        return "Please specify the fields to extract."
    try:
        collection = _build_collection_from_pdf(file_path)
        search_query = query.strip() if query and query.strip() else fields
        results = collection.query(query_texts=[search_query], n_results=5)
        context_docs = results["documents"][0]

        prompt = (
            "You are an information extraction system.\n"
            "Extract ONLY the following fields from the context below.\n"
            "Return the result as valid JSON.\n\n"
            "Fields to extract: " + fields + "\n\n"
            "Context:\n" + "\n".join(context_docs)
        )
        return _call_llm(prompt)
    except Exception as exc:
        return f"Error: {exc}"


def predict_fn(text: str) -> str:
    if not text or not text.strip():
        return "Please enter some text."
    try:
        prediction = predict_text(text)
        return f"Result: {prediction.upper()}"
    except Exception as exc:
        return f"Error: {exc}"


def chat_fn(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."
    prompt = (
        "You are a helpful, friendly assistant. "
        "Be concise, clear, and practical.\n\n"
        "Question: " + question.strip()
    )
    return _call_llm(prompt)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Document AI Demo") as demo:
    gr.Markdown("# Document AI Demo")

    with gr.Tab("OCR"):
        gr.Markdown("### Extract text from an image or PDF")
        with gr.Row():
            ocr_file = gr.File(label="Upload image / PDF", file_types=[".png", ".jpg", ".jpeg", ".pdf"], type="filepath")
            ocr_lang = gr.Textbox(label="Language code", value="eng")
        ocr_btn = gr.Button("Run OCR", variant="primary")
        ocr_out = gr.Textbox(label="Extracted text", lines=12)
        ocr_btn.click(ocr_fn, inputs=[ocr_file, ocr_lang], outputs=ocr_out)

    with gr.Tab("RAG Q&A"):
        gr.Markdown("### Ask a question answered from a PDF")
        rag_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        rag_q = gr.Textbox(label="Question")
        rag_btn = gr.Button("Ask", variant="primary")
        rag_out = gr.Textbox(label="Answer", lines=10)
        rag_btn.click(rag_fn, inputs=[rag_file, rag_q], outputs=rag_out)

    with gr.Tab("Extract Fields"):
        gr.Markdown("### Extract specific fields from a PDF")
        ext_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        ext_query = gr.Textbox(label="Query / context")
        ext_fields = gr.Textbox(label="Fields to extract (comma-separated)", placeholder="name, date, amount")
        ext_btn = gr.Button("Extract", variant="primary")
        ext_out = gr.Textbox(label="Result", lines=10)
        ext_btn.click(extract_fn, inputs=[ext_file, ext_query, ext_fields], outputs=ext_out)

    with gr.Tab("ML Predict"):
        gr.Markdown("### Spam / Ham classifier")
        pred_text = gr.Textbox(label="Input text", lines=5)
        pred_btn = gr.Button("Predict", variant="primary")
        pred_out = gr.Textbox(label="Result")
        pred_btn.click(predict_fn, inputs=pred_text, outputs=pred_out)

    with gr.Tab("Chat"):
        gr.Markdown("### General LLM assistant")
        chat_q = gr.Textbox(label="Your question", lines=4)
        chat_btn = gr.Button("Send", variant="primary")
        chat_out = gr.Textbox(label="Response", lines=10)
        chat_btn.click(chat_fn, inputs=chat_q, outputs=chat_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
