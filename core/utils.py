# core/utils.py
import os

import pytesseract
from bs4 import BeautifulSoup
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
from docx import Document
import hashlib

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
        print("Loading EasyOCR from local storage...")
        ocrReader = easyocr.Reader(
            ['ar', 'en'],
            gpu=False,
        )
        print("EasyOCR finished loading.")

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
            # Clean error message from non-ASCII characters to prevent Windows encoding crashes
            clean_error = "".join([c for c in str(e) if ord(c) < 128])
            raise ValueError(f"EasyOCR failed for image {file_path}: {clean_error}")

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
# ==========================
# Hash Function
# ==========================

def _hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

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





##############agent3################

def detect_language(text):
    return detect(text)


def normalize_text_for_embedding(text):
    """Aggressive normalization for better embedding/clustering: lowercase, no special chars, no extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_embedding(model, text):
    normalized = normalize_text_for_embedding(text)
    return model.encode(normalized).tolist()


def normalize_text_senior(text):
    """Senior-level text cleaning: strips URLs, paths, duplicates, and artifacts."""
    if not text:
        return ""
    
    # 1. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 2. Remove system paths (e.g., C:\Users\...)
    text = re.sub(r'[a-zA-Z]:\\[\\\w\s.-]+', ' ', text)
    
    # 3. OCR artifact stripping
    text = re.sub(r'[●○▲▼■´ˆ]', '', text)
    
    # 4. Handle duplicated tokens (e.g., "Big Data Big Data" or "spum spum")
    # This regex compresses adjacent identical words (case-insensitive)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_sentiment_triggered(text, trigger_dict):
    """Checks if text contains opinion signatures to trigger sentiment analysis."""
    text_lower = text.lower()
    for lang, keywords in trigger_dict.items():
        if any(f" {kw} " in f" {text_lower} " for kw in keywords):
            return True
    return False

def extract_entities(text, language, ner_en, ner_fr, valid_labels=None, tech_keywords=None, generic_blacklist=None, org_verbs=None):
    if language.startswith("fr"):
        doc = ner_fr(text)
    else:
        doc = ner_en(text)

    entities = {} 
    tech_keywords = tech_keywords or []
    generic_blacklist = generic_blacklist or []
    org_verbs = org_verbs or []

    for ent in doc.ents:
        t = ent.text.strip()
        label = ent.label_
        
        # 1. Global Filters
        if len(t) < 3 or t.islower():
            continue
        if any(sym in t for sym in ["/", "http", "="]):
            continue
        if any(gb.lower() == t.lower() for gb in generic_blacklist):
            continue
            
        # 2. TECH DETECTION (Priority 1)
        is_tech = False
        entity_words = [w.lower() for w in t.split()]
        
        # Check if entire entity or ANY word in it is a tech keyword
        if any(tk.lower() == t.lower() for tk in tech_keywords):
            is_tech = True
        elif any(tk.lower() in entity_words for tk in tech_keywords):
            is_tech = True
        elif t.isupper() and 2 <= len(t) <= 6 and t.isalpha(): # Acronyms (Only letters)
            is_tech = True
        elif re.search(r'^[a-zA-Z]+-?\d+(\.\d+)*$', t): # Versioned
            is_tech = True
            
        if is_tech:
            entities[t] = "TECH"
            continue
            
        # 3. Label-Specific Rules
        if label == "PERSON":
            # Reject if contains digits or symbols that shouldn't be in a name
            if any(char.isdigit() for char in t) or "@" in t or "/" in t or "\\" in t:
                continue
            
            # Reject if contains weird symbols like ٤, ٧, etc.
            if re.search(r'[^\w\s\.\-\']', t):
                continue
                
            words = t.split()
            # Strict: 2+ words, each starts with Uppercase
            if len(words) < 2 or not all(w[0].isupper() for w in words if w):
                continue
            
            # Reject if ANY word in the person name is in the generic blacklist
            # This catches "Lien Foreclosure", "Accountant Performed", etc.
            if any(w.lower() in [gb.lower() for gb in generic_blacklist] for w in words):
                continue

            # Reject garbage OCR names (e.g., Aacnnio)
            # Check for high consonant density or very long words without vowels
            for word in words:
                vowels = sum(1 for c in word.lower() if c in "aeiouy")
                if len(word) > 4 and vowels == 0:
                    is_human = False
                    break
                if len(word) > 10 and vowels / len(word) < 0.15: # Too few vowels
                    is_human = False
                    break
            else:
                is_human = True
            
            if not is_human:
                continue
                
        elif label == "ORG":
            # Reclassify or Skip if contains action verbs (it's likely a sentence fragment)
            if any(v.lower() in t.lower() for v in org_verbs):
                continue
            # Skip if too many words (usually a sentence, not an ORG name)
            if len(t.split()) > 5:
                continue
            if any(gb.lower() in t.lower() for gb in generic_blacklist):
                continue
        
        # Priority Check: Only add if not already marked as TECH
        if t not in entities:
            # Final check against blacklist for substrings
            if not any(gb.lower() == t.lower() for gb in generic_blacklist):
                entities[t] = label
    
    # PRODUCTION FORMAT: List of Objects
    return [{"text": text, "label": label} for text, label in entities.items()]


def analyze_sentiment(model, text, trigger_dict=None):
    # Trigger Logic (Production Version)
    if trigger_dict and not is_sentiment_triggered(text, trigger_dict):
        return "NEUTRAL"
        
    try:
        result = model(text)[0]
        return result["label"]
    except:
        return "NEUTRAL"


def detect_pii(engine, text, language, threshold, important_pii=None, tech_keywords=None):
    results = engine.analyze(
        text=text,
        language=language,
        score_threshold=max(threshold, 0.75) 
    )
    
    pii_found = {} # Dedup by text
    tech_keywords = tech_keywords or []

    for r in results:
        etype = r.entity_type
        if etype == "EMAIL_ADDRESS": etype = "EMAIL"
            
        if important_pii and etype not in important_pii:
            continue
            
        actual_text = text[r.start:r.end].strip().strip(".,!?:;")
        
        # PRODUCTION RULES for PII Names
        if etype == "PERSON":
            words = actual_text.split()
            if len(words) < 2 or not all(w[0].isupper() for w in words if w):
                continue
        
        # Block TECH and single Uppercase tokens
        if any(tk.lower() == actual_text.lower() for tk in tech_keywords):
            continue
        if actual_text.isupper() and " " not in actual_text:
            continue
            
        # Mapping to user-friendly types
        etype_map = {
            "PERSON": "PERSON",
            "LOCATION": "LOCATION",
            "EMAIL": "EMAIL",
            "PHONE_NUMBER": "PHONE"
        }
        final_type = etype_map.get(etype, "OTHER")

        pii_found[actual_text] = {
            "text": actual_text,
            "type": final_type
        }
        
    # PRODUCTION FORMAT: List of Objects
    return list(pii_found.values())


def analyze_sentiment_batch(model, sentences):
    """
    Analyze sentiment for a list of sentences (batch mode).
    Returns list of labels.
    """
    results = model(
    sentences,
    truncation=True,
    max_length=512
)
    return [r["label"] for r in results]




