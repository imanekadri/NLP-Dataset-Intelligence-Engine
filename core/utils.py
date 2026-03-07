# core/utils.py
import os

import pytesseract
from bs4 import BeautifulSoup
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
from docx import Document

import json
import re
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
from langdetect import detect, DetectorFactory
import easyocr

import numpy as np



############### agent1  #############
# Configure Tesseract OCR path if provided via env or on Windows
if os.name == 'nt':
    # default Windows install path (override with TESSERACT_CMD env var if needed)
    pytesseract.pytesseract.tesseract_cmd = os.environ.get(
        'TESSERACT_CMD', r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    # on Linux/macOS, allow override but otherwise expect tesseract on PATH
    if os.environ.get('TESSERACT_CMD'):
        pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD')

ocrReader = None

def get_reader():
    global ocrReader
    if ocrReader is None:
        print("🚀 Loading EasyOCR from local storage...")
        ocrReader = easyocr.Reader(
            ['ar', 'en'],
            gpu=False,
        )
        print("✅ EasyOCR finished loading.")

    return ocrReader
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

        text = ""

        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
                text = "\n".join(pages_text)
        except Exception:
            text = ""

        # If no text extracted → use OCR
        if not text.strip():
            try:
                poppler_path = os.environ.get('POPPLER_PATH')
                images = convert_from_path(file_path, poppler_path=poppler_path)

                ocr_pages = []
                ocrReader = get_reader()  # load once

                for img in images:
                    img_np = np.array(img)
                    page_text = ocrReader.readtext(img_np, detail=0)
                    ocr_pages.append("\n".join(page_text))

                text = "\n".join(ocr_pages)

            except Exception as e:
                raise ValueError(f"OCR failed for PDF {file_path}: {e}")

        return text

    # Image files (OCR)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        try:
            # 2. EasyOCR's readtext can take a file path directly
            # detail=0 returns ONLY the text strings, making extraction easy
            ocrReader = get_reader()
            result = ocrReader.readtext(file_path, detail=0)

            if not result:
                return ""

            # 3. Join the detected text lines with newlines
            return "\n".join(result)

        except Exception as e:
            # It's helpful to know exactly which file failed
            raise ValueError(f"EasyOCR failed for image {file_path}: {str(e)}")

    # DOCX files
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # HTML files
    elif ext in [".html", ".htm"]:
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    raw = f.read()
                soup = BeautifulSoup(raw, "lxml")
                for s in soup(["script", "style"]):
                    s.extract()
                text = soup.get_text(separator="\n", strip=True)
                return text
            except Exception:
                continue
        raise ValueError(f"Cannot read html file {file_path} with any encoding")

    else:  # fallback
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise ValueError(f"Cannot read file {file_path} with any encoding")


def extract_html_metadata(file_path, encodings_to_try):
    """Return metadata from an HTML file: title, meta description, headings, links, json-ld."""
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                raw = f.read()
            soup = BeautifulSoup(raw, "lxml")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            meta_desc = ""
            m = soup.find("meta", attrs={"name": "description"})
            if m and m.get("content"):
                meta_desc = m.get("content").strip()

            headings = []
            for h in soup.find_all(["h1", "h2", "h3"]):
                if h.text:
                    headings.append(h.text.strip())

            links = []
            for a in soup.find_all("a", href=True):
                links.append(a["href"])

            json_ld = []
            for s in soup.find_all("script", type=lambda v: v and "application/ld+json" in v):
                try:
                    payload = json.loads(s.string) if s.string else None
                except Exception:
                    payload = None
                if payload:
                    json_ld.append(payload)

            return {
                "title": title,
                "meta_description": meta_desc,
                "headings": headings,
                "links": links,
                "json_ld": json_ld,
            }
        except Exception:
            continue
    return {}


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


def calculate_noise_ratio(text):
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    return special_chars / len(text) if len(text) > 0 else 0


def detect_structure(text):
    dialogue_pattern = r'(^|\n)(User:|Assistant:|Q:|A:|Speaker)'
    code_pattern = r'(def |class |#include|console\.log|public static|import )'

    dialogue = re.search(dialogue_pattern, text)
    code = re.search(code_pattern, text)

    return bool(dialogue), bool(code)


def lexical_diversity(text):
    words = text.split()
    if not words:
        return 0
    return len(set(words)) / len(words)


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



    #     if not text.strip():
    #         try:
    #             # convert PDF pages to images; on Linux/macOS poppler should be on PATH
    #             poppler_path = os.environ.get('POPPLER_PATH')
    #             if poppler_path:
    #                 images = convert_from_path(file_path, poppler_path=poppler_path)
    #             else:
    #                 images = convert_from_path(file_path)
    #             ocr_texts = [pytesseract.image_to_string(img) for img in images]
    #             text = "\n".join(ocr_texts)
    #         except Exception as e:
    #             raise ValueError(f"OCR failed for {file_path}: {e}")
    #     return text
    #
    # # Image files (OCR)
    # elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
    #     try:
    #         img = Image.open(file_path)
    #         return pytesseract.image_to_string(img)
    #     except Exception as e:
    #         raise ValueError(f"OCR failed for image {file_path}: {e}")
