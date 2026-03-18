import os
import json
import re
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import ollama
from sklearn.feature_extraction.text import HashingVectorizer


DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
RAG_MODEL_NAME = os.getenv("RAG_LLM_MODEL", os.getenv("GENERAL_LLM_MODEL", "tinyllama"))
COLLECTION_NAME = "documents"
EMBED_DIM = 512
_EMBEDDER = HashingVectorizer(
    n_features=EMBED_DIM,
    alternate_sign=False,
    norm="l2",
)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    clean_texts = [t if isinstance(t, str) else "" for t in texts]
    matrix = _EMBEDDER.transform(clean_texts)
    return matrix.toarray().tolist()


def _load_documents(file_path: str | None = None):
    if file_path:
        loader = PyPDFLoader(file_path)
        return loader.load()

    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def build_vector_db(file_path: str | None = None):

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Always rebuild from the current file(s) to avoid stale retrieval from old uploads.
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    raw_documents = _load_documents(file_path=file_path)
    if not raw_documents:
        raise ValueError("No PDF content found to index.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(raw_documents)

    documents = [chunk.page_content for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]

    if not documents:
        raise ValueError("No text chunks could be created from the document.")

    if file_path:
        file_tag = os.path.basename(file_path)
    else:
        file_tag = "all_docs"
    ids = [f"{file_tag}_ID{i}" for i in range(len(chunks))]

    batch_size = 50

    for i in range(0, len(documents), batch_size):
        docs_batch = documents[i:i + batch_size]
        collection.upsert(
            documents=docs_batch,
            embeddings=_embed_texts(docs_batch),
            metadatas=metadata[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    return collection


def retrieve_docs(collection, query):

    results = collection.query(
        query_embeddings=_embed_texts([query]),
        n_results=4
    )

    docs: List[str] = results.get("documents", [[]])[0]
    return docs


def _heuristic_extract(context_text: str, fields: List[str]) -> dict:
    extracted: dict = {}
    for field in fields:
        safe = re.escape(field)
        # Prefer line-based "Field: value"
        line_match = re.search(rf"(?im)^\s*{safe}\s*[:\-]\s*(.+?)\s*$", context_text)
        if line_match:
            extracted[field] = line_match.group(1).strip()
            continue

        # Fallback for inline "Field: value" patterns.
        inline_match = re.search(rf"(?i){safe}\s*[:\-]\s*([^\n,;]+)", context_text)
        if inline_match:
            extracted[field] = inline_match.group(1).strip()
            continue

        extracted[field] = "Not found"

    return extracted





def ask_llm(context_docs, question):
    if not context_docs:
        return "No relevant context was found in the uploaded document."
    context_text = "\n\n".join(context_docs) if isinstance(context_docs, list) else str(context_docs)

    prompt = f"""
     You are a helpful assistant.
     Answer only from the provided context.
     If the answer is not in context, say: "Not found in provided document."
     Keep the answer concise.

     Context:
     {context_text}

     Question: {question}
     """

    try:
        response = ollama.chat(
            model=RAG_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 256},
        )
        return response["message"]["content"]
    except Exception as exc:
        context_preview = "\n".join(context_docs[:2]) if isinstance(context_docs, list) else str(context_docs)
        return (
            "RAG fallback response: Ollama is unavailable. "
            f"Start Ollama to get full LLM answers (model: {RAG_MODEL_NAME}). Error: {exc}\n\n"
            f"Retrieved context preview:\n{context_preview[:1200]}"
        )


def extract_information(context_docs, fields):
    if not context_docs:
        return {"error": "No relevant context was found in the uploaded document."}
    context_text = "\n\n".join(context_docs) if isinstance(context_docs, list) else str(context_docs)

    prompt = f"""
    You are an information extraction system.

    Extract ONLY the following fields from the context.
    Return the result as JSON.
    If a field is missing, set it to "Not found".

    Fields:
    {fields}

    Context:
    {context_text}
    """

    try:
        response = ollama.chat(
            model=RAG_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 256},
        )
        content = response["message"]["content"].strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                for field, value in _heuristic_extract(context_text, fields).items():
                    parsed.setdefault(field, value)
                return parsed
            return {"result": parsed}
        except json.JSONDecodeError:
            return {
                "warning": "LLM did not return valid JSON; heuristic extraction used.",
                "fields": _heuristic_extract(context_text, fields),
                "raw_response": content,
            }
    except Exception as exc:
        return {
            "warning": f"Ollama unavailable, fallback extraction used: {exc}",
            "fields": _heuristic_extract(context_text, fields),
        }
