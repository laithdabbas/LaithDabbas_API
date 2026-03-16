from fastapi import FastAPI, UploadFile, File, Form
import shutil, os
import sys
from pydantic import BaseModel
from app.services.ML_Model import predict_text
from app.services.ocr_utils import run_ocr
from app.services.rag import DATA_PATH, build_vector_db, extract_information, retrieve_docs, ask_llm
from app.services.model import ask_general_llm
import gradio as gr

# =========================
# FASTAPI SETUP
# =========================

class ExtractionInput(BaseModel):
    query: str
    fields: list[str]

class TextInput(BaseModel):
    text: str

class ChatInput(BaseModel):
    question: str

app = FastAPI()

def _runtime_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = _runtime_base_dir()

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

@app.get("/")
def home():
    return {"message": "OCR + RAG API running"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...), lang: str = Form("eng")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = run_ocr(file_path, lang)
    os.remove(file_path)
    return {"filename": file.filename, "ocr_text": text}

@app.post("/RAG")
async def rag(file: UploadFile = File(...), question: str = Form(...)):
    file_path = os.path.join(DATA_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    collection = build_vector_db()
    docs = retrieve_docs(collection, question)
    answer = ask_llm(docs, question)
    return {"filename": file.filename, "question": question, "answer": answer}

@app.post("/extract")
def extract(file: UploadFile = File(...), query: str = Form(...), fields: str = Form(...)):
    file_path = os.path.join(DATA_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    collection = build_vector_db()
    docs = retrieve_docs(collection, query)
    fields_list = [f.strip() for f in fields.split(",") if f.strip()]
    extracted = extract_information(docs, fields_list)
    return {"query": query, "fields": fields_list, "extracted_information": extracted}

@app.post("/predict")
def predict(data: TextInput):
    result = predict_text(data.text)
    return {"input": data.text, "prediction": result}

@app.post("/chat")
def chat(data: ChatInput):
    answer = ask_general_llm(data.question)
    return {"question": data.question, "answer": answer}

# =========================
# GRADIO UI
# =========================

def ocr_ui(file_path, lang):
    return run_ocr(file_path, lang)

def rag_ui(file_path, question):
    collection = build_vector_db()
    docs = retrieve_docs(collection, question)
    return ask_llm(docs, question)

def extract_ui(file_path, query, fields):
    collection = build_vector_db()
    docs = retrieve_docs(collection, query)
    fields_list = [f.strip() for f in fields.split(",") if f.strip()]
    return extract_information(docs, fields_list)

def predict_ui(text):
    return predict_text(text.strip())

def chat_ui(question):
    return ask_general_llm(question.strip())

with gr.Blocks() as demo:
    gr.Markdown("# Document AI Demo")

    with gr.Tab("OCR"):
        file_input = gr.File(label="Upload PDF/Image", type="filepath")
        lang_input = gr.Textbox(label="Language", value="eng")
        output_text = gr.Textbox(label="OCR Text")
        btn = gr.Button("Run OCR")
        btn.click(fn=ocr_ui, inputs=[file_input, lang_input], outputs=[output_text])

    with gr.Tab("RAG QA"):
        file_input2 = gr.File(label="Upload PDF", type="filepath")
        question_input = gr.Textbox(label="Question")
        output_answer = gr.Textbox(label="Answer")
        btn = gr.Button("Ask Question")
        btn.click(fn=rag_ui, inputs=[file_input2, question_input], outputs=[output_answer])

    with gr.Tab("Extraction"):
        file_input3 = gr.File(label="Upload PDF", type="filepath")
        query_input = gr.Textbox(label="Query")
        fields_input = gr.Textbox(label="Fields (comma-separated)")
        output_extract = gr.JSON(label="Extracted Info")
        btn = gr.Button("Extract")
        btn.click(fn=extract_ui, inputs=[file_input3, query_input, fields_input], outputs=[output_extract])

    with gr.Tab("Prediction"):
        predict_input = gr.Textbox(label="Input Text", lines=4)
        predict_output = gr.Textbox(label="Prediction")
        btn = gr.Button("Predict")
        btn.click(fn=predict_ui, inputs=[predict_input], outputs=[predict_output])

    with gr.Tab("General Assistant"):
        chat_input = gr.Textbox(label="Ask Anything", lines=4)
        chat_output = gr.Textbox(label="Assistant Answer", lines=8)
        btn = gr.Button("Ask Assistant")
        btn.click(fn=chat_ui, inputs=[chat_input], outputs=[chat_output])

# mount Gradio inside FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)