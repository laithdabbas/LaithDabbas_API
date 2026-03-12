from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from pydantic import BaseModel
from app.services.ML_Model import predict_text   
from app.services.ocr_utils import run_ocr 
from app.services.rag import DATA_PATH, build_vector_db, extract_information, retrieve_docs, ask_llm


class ExtractionInput(BaseModel):
    query: str
    fields: list[str]


class TextInput(BaseModel):
    text: str

app = FastAPI()

UPLOAD_DIR = "uploads"
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

    # Build vector DB
    collection = build_vector_db()

    # Retrieve
    context_docs = retrieve_docs(collection, question)

    # Ask LLM
    answer = ask_llm(context_docs, question)

    return {
        "filename": file.filename,
        "question": question,
        "answer": answer
    }

@app.post("/extract")
def extract(file: UploadFile = File(...), query: str = Form(...), fields: str = Form(...)):

    file_path = os.path.join(DATA_PATH, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)



    collection = build_vector_db()
    docs = retrieve_docs(collection, query)

    fields_list = [f.strip() for f in fields.split(",") if f.strip()]
    extracted = extract_information(docs, fields_list)

    return {
        "query": query,
        "fields": fields_list,
        "extracted_information": extracted
    }

@app.post("/predict")
def predict(data: TextInput):
    result = predict_text(data.text)

    return {
        "input": data.text,
        "prediction": result }
