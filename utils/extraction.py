import os
import pytesseract
import PyPDF2
from pdf2image import convert_from_path
from docx import Document


def init_tesseract(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def extract_text_from_file(file_path, encodings_to_try, poppler_path=None):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".py", ".log", ".js"]:
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read file {file_path} with any encoding")

    elif ext in [".csv", ".json", ".xml"]:
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read structured file {file_path} with any encoding")

    elif ext == ".pdf":
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
                text = "\n".join(pages_text)
        except Exception:
            text = ""

        if not text.strip():
            try:
                images = convert_from_path(file_path, poppler_path=poppler_path)
                ocr_texts = [pytesseract.image_to_string(img) for img in images]
                text = "\n".join(ocr_texts)
            except Exception as e:
                raise ValueError(f"OCR failed for {file_path}: {e}")
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read file {file_path} with any encoding")
