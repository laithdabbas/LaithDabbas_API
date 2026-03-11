import gradio as gr
from app.services.ocr_utils import run_ocr
from app.services.rag import retrieve_docs, ask_llm, extract_information, build_vector_db

# Build vector DB once
collection = build_vector_db()

def ocr_demo(file):
    text = run_ocr(file.name)
    return text

def rag_demo(file, question):
    docs = retrieve_docs(collection, question)
    answer = ask_llm(docs, question)
    return answer

def extract_demo(file, query, fields):
    docs = retrieve_docs(collection, query)
    extracted = extract_information(docs, fields.split(","))
    return extracted

with gr.Blocks() as demo:
    gr.Markdown("# Document AI Demo")
    
    with gr.Tab("OCR"):
        file_input = gr.File(label="Upload PDF/Image")
        output_text = gr.Textbox(label="OCR Text")
        gr.Button("Run OCR").click(ocr_demo, inputs=file_input, outputs=output_text)
    
    with gr.Tab("RAG QA"):
        file_input2 = gr.File(label="Upload PDF")
        question_input = gr.Textbox(label="Question")
        output_answer = gr.Textbox(label="Answer")
        gr.Button("Ask Question").click(rag_demo, inputs=[file_input2, question_input], outputs=output_answer)
    
    with gr.Tab("Extraction"):
        file_input3 = gr.File(label="Upload PDF")
        query_input = gr.Textbox(label="Query")
        fields_input = gr.Textbox(label="Fields (comma-separated)")
        output_extract = gr.JSON(label="Extracted Info")
        gr.Button("Extract").click(extract_demo, inputs=[file_input3, query_input, fields_input], outputs=output_extract)

demo.launch()