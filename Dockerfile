FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (required for pytesseract and pdf2image)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_api.py .
COPY app/ ./app/

EXPOSE 8000 7860

CMD ["uvicorn", "run_api:app", "--host", "0.0.0.0", "--port", "8000"]

