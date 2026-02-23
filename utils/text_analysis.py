import os
import json
import re
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


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
                print(f"[text_analysis] Error reading {file}: {e}")
    return texts


def calculate_noise_ratio(text):
    if not text:
        return 0.0
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    return special_chars / len(text)


def detect_structure(text):
    if not text:
        return "Paragraph"
    if any(kw in text for kw in ["def ", "import ", "function", "{", "}", "class ", "return "]):
        return "Code"
    elif text.count(":") > len(text.split(".")) and len(text) < 1000:
        return "Dialogue"
    else:
        return "Paragraph"


def detect_languages(texts, sample_size=200):
    lang_counter = Counter()
    for text in texts[:sample_size]:
        if not text or len(text.strip()) < 10:
            continue
        try:
            lang = detect(text)
            lang_counter[lang] += 1
        except Exception:
            continue
    return dict(lang_counter)
