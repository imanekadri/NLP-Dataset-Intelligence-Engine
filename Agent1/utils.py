import os
import pytesseract
import PyPDF2
from pdf2image import convert_from_path
from docx import Document
import json
import csv

# Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_file(file_path, encodings_to_try):
    """
    Extract text from any supported file:
    - For txt, py, log, js → read as text
    - For csv, json, xml → keep structure but return as string
    - For pdf → try digital extraction first, then OCR if empty
    - For docx → extract paragraphs
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    
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
                    return f.read()  # keep structure
            except Exception:
                continue
        raise ValueError(f"Cannot read structured file {file_path} with any encoding")

    # ملفات PDF
    elif ext == ".pdf":
        # pdf file 
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
                text = "\n".join(pages_text)
        except Exception:
            text = ""

        # if the pdf is empty then use OCR
        if not text.strip():
            try:
                images = convert_from_path(file_path, poppler_path=r"C:\Program Files\poppler-25.12.0\Library\bin")
                ocr_texts = [pytesseract.image_to_string(img) for img in images]
                text = "\n".join(ocr_texts)
            except Exception as e:
                raise ValueError(f"OCR failed for {file_path}: {e}")
        return text

    # docx
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # fallback
    else:
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read file {file_path} with any encoding")