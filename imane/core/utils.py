# core/utils.py
import os
import pytesseract
import PyPDF2
from pdf2image import convert_from_path
from docx import Document

import json
import re
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
from langdetect import detect, DetectorFactory


############### agent1  #############
# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_file(file_path, encodings_to_try):
    """Extract text from any supported file type"""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    # Plain text files
    if ext in [".txt", ".py", ".log", ".js"]:
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read file {file_path} with any encoding")

    # Structured files
    elif ext in [".csv", ".json", ".xml"]:
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read structured file {file_path} with any encoding")

    # PDF files
    elif ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
                text = "\n".join(pages_text)
        except Exception:
            text = ""

        if not text.strip():
            try:
                images = convert_from_path(file_path, poppler_path=r"C:\Program Files\poppler-25.12.0\Library\bin")
                ocr_texts = [pytesseract.image_to_string(img) for img in images]
                text = "\n".join(ocr_texts)
            except Exception as e:
                raise ValueError(f"OCR failed for {file_path}: {e}")
        return text

    # DOCX files
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:  # fallback
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read file {file_path} with any encoding")
    


########## agent2 ###################

DetectorFactory.seed = 0


# Load Text Files
def load_texts_from_folder(folder_path):
    texts = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)

            try:
                if file.endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        texts.append(f.read())

                elif file.endswith(".csv"):
                    df = pd.read_csv(path)
                    for col in df.columns:
                        texts.extend(df[col].astype(str).tolist())

                elif file.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    texts.extend([str(v) for v in item.values()])
                        elif isinstance(data, dict):
                            texts.extend([str(v) for v in data.values()])

                elif file.endswith(".xml"):
                    tree = ET.parse(path)
                    root_xml = tree.getroot()
                    for elem in root_xml.iter():
                        if elem.text:
                            texts.append(elem.text)

            except Exception as e:
                print(f"Error reading {file}: {e}")

    return texts


# Noise Detection
def calculate_noise_ratio(text):
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    return special_chars / len(text) if len(text) > 0 else 0


# Structure Detection
def detect_structure(text):
    if any(kw in text for kw in ["def ", "import ", "function", "{", "}"]):
        return "Code"

    elif text.count(":") > len(text.split(".")) and len(text) < 1000:
        return "Dialogue"

    else:
        return "Paragraph"


# Language Detection
def detect_languages(texts, sample_size):
    lang_counter = Counter()

    for text in texts[:sample_size]:
        try:
            lang = detect(text)
            lang_counter[lang] += 1
        except:
            continue

    return dict(lang_counter)