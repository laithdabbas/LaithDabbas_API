import os
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _require_file(file_path: str):
    if not file_path:
        raise gr.Error("Please upload a file first.")


def _post(endpoint: str, *, files=None, data=None, json=None):
    url = f"{API_URL}{endpoint}"
    try:
        response = requests.post(url, files=files, data=data, json=json, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        if hasattr(exc, "response") and exc.response is not None:
            try:
                detail = exc.response.json()
            except ValueError:
                detail = exc.response.text
        raise gr.Error(f"API request failed: {detail or str(exc)}")
    return response.json()


def ocr_demo(file_path, lang):
    _require_file(file_path)
    with open(file_path, "rb") as handle:
        payload = _post(
            "/ocr",
            files={"file": (os.path.basename(file_path), handle)},
            data={"lang": lang},
        )
    return payload.get("ocr_text", "")


def rag_demo(file_path, question):
    _require_file(file_path)
    if not question:
        raise gr.Error("Please enter a question.")
    with open(file_path, "rb") as handle:
        payload = _post(
            "/RAG",
            files={"file": (os.path.basename(file_path), handle)},
            data={"question": question},
        )
    return payload.get("answer", "")


def extract_demo(file_path, query, fields):
    _require_file(file_path)
    if not query:
        raise gr.Error("Please enter a query.")
    with open(file_path, "rb") as handle:
        payload = _post(
            "/extract",
            files={"file": (os.path.basename(file_path), handle)},
            data={"query": query, "fields": fields or ""},
        )
    return payload.get("extracted_information", "")


def predict_demo(text):
    if not text or not text.strip():
        raise gr.Error("Please enter text for prediction.")
    payload = _post("/predict", json={"text": text.strip()})
    return payload.get("prediction", "")


def chat_demo(question):
    if not question or not question.strip():
        raise gr.Error("Please enter a question.")
    payload = _post("/chat", json={"question": question.strip()})
    return payload.get("answer", "")


with gr.Blocks() as demo:
    gr.Markdown("# Document AI Demo")
    gr.Markdown(f"**API URL:** `{API_URL}`")

    with gr.Tab("OCR"):
        file_input = gr.File(label="Upload PDF/Image", type="filepath")
        lang_input = gr.Textbox(label="Language", value="eng")
        output_text = gr.Textbox(label="OCR Text")
        gr.Button("Run OCR").click(ocr_demo, inputs=[file_input, lang_input], outputs=output_text)

    with gr.Tab("RAG QA"):
        file_input2 = gr.File(label="Upload PDF", type="filepath")
        question_input = gr.Textbox(label="Question")
        output_answer = gr.Textbox(label="Answer")
        gr.Button("Ask Question").click(
            rag_demo, inputs=[file_input2, question_input], outputs=output_answer
        )

    with gr.Tab("Extraction"):
        file_input3 = gr.File(label="Upload PDF", type="filepath")
        query_input = gr.Textbox(label="Query")
        fields_input = gr.Textbox(label="Fields (comma-separated)")
        output_extract = gr.JSON(label="Extracted Info")
        gr.Button("Extract").click(
            extract_demo, inputs=[file_input3, query_input, fields_input], outputs=output_extract
        )

    with gr.Tab("Prediction"):
        predict_input = gr.Textbox(label="Input Text", lines=4)
        predict_output = gr.Textbox(label="Prediction")
        gr.Button("Predict").click(
            predict_demo, inputs=[predict_input], outputs=predict_output
        )

    with gr.Tab("General Assistant"):
        chat_input = gr.Textbox(label="Ask Anything", lines=4)
        chat_output = gr.Textbox(label="Assistant Answer", lines=8)
        gr.Button("Ask Assistant").click(
            chat_demo, inputs=[chat_input], outputs=chat_output
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
