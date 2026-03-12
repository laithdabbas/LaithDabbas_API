import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POPPLER_PATH = os.path.join(BASE_DIR, "poppler_dir", "poppler-24.08.0", "Library", "bin")

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image_cv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh

def run_ocr_image(file_path, language="eng"):
    image = cv2.imread(file_path)
    processed = preprocess_image_cv(image)
    text = pytesseract.image_to_string(Image.fromarray(processed), lang=language)
    return text

def preprocess_pdf(pdf_path):
    if os.name == "nt":
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    else:
        pages = convert_from_path(pdf_path, dpi=300)
    processed_pages = []
    for page in pages:
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        processed_pages.append(thresh)
    return processed_pages

def run_ocr_pdf(file_path, language="eng"):
    pages = preprocess_pdf(file_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(Image.fromarray(page), lang=language)
    return text

def run_ocr(file_path, language="eng"):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".pdf"]:
        return run_ocr_pdf(file_path, language)
    else:
        return run_ocr_image(file_path, language)
