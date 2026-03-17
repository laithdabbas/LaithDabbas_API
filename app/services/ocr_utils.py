import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader
import pytesseract

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POPPLER_PATH = os.path.join(BASE_DIR, "poppler_dir", "poppler-24.08.0", "Library", "bin")
POPPLER_PATH = os.getenv("POPPLER_PATH", DEFAULT_POPPLER_PATH)

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

def _read_image(file_path: str):
    """Read image bytes in a way that works with unicode paths on Windows."""
    image = None
    try:
        # np.fromfile handles unicode paths better on Windows.
        data = np.fromfile(file_path, dtype=np.uint8)
        if data.size > 0:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        image = None

    if image is None:
        image = cv2.imread(file_path)

    if image is None:
        size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        raise ValueError(
            f"Could not read image file '{file_path}'. "
            f"File size: {size} bytes. Supported formats: PNG/JPG."
        )
    return image

def run_ocr_image(file_path, language="eng"):
    image = _read_image(file_path)
    processed = preprocess_image_cv(image)
    text = pytesseract.image_to_string(Image.fromarray(processed), lang=language)
    return text

def preprocess_pdf(pdf_path):
    if os.name == "nt":
        pdfinfo_exe = os.path.join(POPPLER_PATH, "pdfinfo.exe")
        if not os.path.exists(pdfinfo_exe):
            raise ValueError(
                "Poppler not found. Set POPPLER_PATH to the folder containing pdfinfo.exe "
                f"(current POPPLER_PATH: '{POPPLER_PATH}')."
            )
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
    try:
        pages = preprocess_pdf(file_path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(Image.fromarray(page), lang=language)
        return text
    except ValueError:
        # Poppler missing. Try text extraction for text-based PDFs as a fallback.
        reader = PdfReader(file_path)
        extracted = "".join(page.extract_text() or "" for page in reader.pages)
        if extracted.strip():
            return extracted
        raise

def run_ocr(file_path, language="eng"):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".pdf"]:
        return run_ocr_pdf(file_path, language)
    else:
        return run_ocr_image(file_path, language)
