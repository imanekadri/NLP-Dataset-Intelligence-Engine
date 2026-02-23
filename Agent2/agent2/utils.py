# utils.py

import os
import json
import re
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
from langdetect import detect, DetectorFactory

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