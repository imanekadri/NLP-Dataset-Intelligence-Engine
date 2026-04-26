"""
Agent 1 - Cleaning : Nettoyage général et robuste des données textuelles.

Ce module applique un pipeline de nettoyage complet AVANT que les données
ne soient envoyées aux agents suivants (Agent 2 profiling, Agent 3 sentiment).

Règle importante : PAS de .lower() → cela détruit la détection NER/PII dans Agent 3.
"""

import os
import re
import csv
import unicodedata
import spacy
from collections import Counter
from core.config import agent1_config as config

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


# ==============================================================
#  Pipeline de nettoyage général
# ==============================================================

def clean_text_pipeline(text: str) -> str:
    """
    Pipeline complet de nettoyage du texte brut.
    Chaque étape est appliquée dans l'ordre pour maximiser la qualité
    du texte envoyé aux agents suivants.
    """
    if not text or not text.strip():
        return ""

    # --- Étape 1 : Réparer les artefacts OCR (lettres espacées) ---
    text = fix_ocr_spaced_characters(text)

    # --- Étape 2 : Supprimer les artefacts de Jupyter Notebook & Pédagogiques ---
    text = remove_notebook_artifacts(text)
    text = remove_pedagogical_noise(text)

    # --- Étape 3 : Supprimer les URLs ---
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # --- Étape 4 : Supprimer les chemins de fichiers Windows/Linux ---
    text = re.sub(r'[a-zA-Z]:\\[\\\w\s.\-]+', ' ', text)
    text = re.sub(r'file\s+\w+\s+\w?Users\b.*?(?=\s{2,}|\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'/(?:home|usr|var|tmp|etc|opt)/\S+', ' ', text)

    # --- Étape 5 : Supprimer les adresses email ---
    text = re.sub(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b', ' [EMAIL] ', text)

    # --- Étape 6 : Supprimer les balises HTML résiduelles ---
    text = re.sub(r'<[^>]+>', ' ', text)

    # --- Étape 7 : Supprimer les en-têtes/pieds de page répétés ---
    text = remove_repeated_headers(text)

    # --- Étape 8 : Supprimer les numéros de page et timestamps ---
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\b', ' ', text)  # 29/09/2025 23:19
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?\b', ' ', text)  # ISO format
    text = re.sub(r'\b\d{1,3}\s*/\s*\d{1,3}\b', ' ', text)  # Page 3/9

    # --- Étape 9 : Supprimer les symboles OCR et caractères spéciaux inutiles ---
    text = re.sub(r'[●○▲▼■□◆◇★☆´ˆ˜¨°±×÷¬¦¡¿©®™€£¥¢]+', ' ', text)
    # Supprimer les caractères de contrôle Unicode (sauf newline/tab)
    text = ''.join(
        ch for ch in text
        if ch in ('\n', '\t', ' ') or not unicodedata.category(ch).startswith('C')
    )

    # --- Étape 10 : Supprimer les mots dupliqués adjacents ---
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # --- Étape 11 : Normaliser les espaces ---
    text = re.sub(r'[ \t]+', ' ', text)          # espaces multiples → un seul
    text = re.sub(r'\n{3,}', '\n\n', text)        # sauts de ligne excessifs
    text = '\n'.join(line.strip() for line in text.split('\n'))  # strip chaque ligne
    text = text.strip()

    # --- Étape 12 : Supprimer les lignes trop courtes (bruit) ---
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.split()) >= 3 or len(line) > 15]
    text = '\n'.join(cleaned_lines)

    return text


def fix_ocr_spaced_characters(text: str) -> str:
    """
    Répare les artefacts OCR où les caractères sont séparés par des espaces.
    Exemple : "m n e K d r" → tentative de reconstruction.
    
    Logique : si on détecte une séquence de lettres isolées (>= 5 lettres seules
    séparées par des espaces), on les fusionne.
    """
    def merge_spaced(match):
        chars = match.group(0).replace(' ', '')
        return chars

    # Détecte 5+ lettres isolées séparées par des espaces : "a b c d e f"
    text = re.sub(
        r'(?<!\w)([a-zA-Z]\s){5,}[a-zA-Z](?!\w)',
        merge_spaced,
        text
    )
    return text


def remove_notebook_artifacts(text: str) -> str:
    """Supprime les artefacts typiques de Jupyter Notebook."""
    text = re.sub(r'In\s*\[\d*\]\s*:', ' ', text)
    text = re.sub(r'Out\s*\[\d*\]\s*:', ' ', text)
    text = re.sub(r'%matplotlib\s+\w+', ' ', text)
    text = re.sub(r'!pip\s+install\s+\S+', ' ', text)
    return text


def remove_pedagogical_noise(text: str) -> str:
    """
    Supprime les bruits liés aux énoncés d'exercices qui faussent le sentiment.
    Ex: 'Problem Statement', 'Exercise 01', 'Tutorial'
    """
    patterns = [
        r'\bProblem\s+Statement\b',
        r'\bExercise\s+\d+\b',
        r'\bExercice\s+\d+\b',
        r'\bTutorial\s+\d+\b',
        r'\bTP\s*\d+\b',
        r'\bTasks?\s*\d*\b',
        r'\bData\s+Initialization\b',
        r'\bBasic\s+Statistics\b'
    ]
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text


def remove_repeated_headers(text: str, threshold: int = 3) -> str:
    """
    Détecte et supprime les lignes qui se répètent trop souvent dans le document
    (typiquement des en-têtes/pieds de page de PDF).
    
    threshold : nombre minimum de répétitions pour considérer une ligne comme header.
    """
    lines = text.split('\n')
    if len(lines) < 10:
        return text

    # Compter les occurrences de chaque ligne (normalisée)
    line_counts = Counter()
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 5:  # ignorer les lignes vides/très courtes
            line_counts[stripped] += 1

    # Identifier les lignes répétées (headers/footers)
    repeated_lines = {line for line, count in line_counts.items() if count >= threshold}

    if not repeated_lines:
        return text

    # Filtrer les lignes répétées (garder seulement la première occurrence)
    seen = set()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in repeated_lines:
            if stripped not in seen:
                seen.add(stripped)
                cleaned_lines.append(line)  # garder la 1ère occurrence
            # sinon on l'ignore
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_stopwords_spacy(text: str) -> str:
    """
    Supprime les stopwords et la ponctuation via spaCy.
    Préserve la casse (PAS de .lower()) pour ne pas casser NER/PII.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc
              if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)


# ==============================================================
#  Fonction principale d'exécution du cleaning
# ==============================================================

def run_cleaning(state):
    """
    Point d'entrée du cleaning Agent 1.
    Lit les fichiers extraits, applique le pipeline de nettoyage,
    sauvegarde les fichiers nettoyés, et met à jour le state + trace CSV.
    """
    print("[Agent1-Cleaning] Start general-purpose cleaning pipeline...")

    os.makedirs(config.CLEANED_DIR, exist_ok=True)

    raw_docs = state.get("raw_docs", [])
    if not raw_docs:
        print("[Agent1-Cleaning] No documents to clean.")
        return state

    cleaned_count = 0
    skipped_count = 0

    for doc in raw_docs:
        try:
            extracted_path = doc.get("extracted_path")
            if not extracted_path or not os.path.exists(extracted_path):
                skipped_count += 1
                continue

            with open(extracted_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                skipped_count += 1
                continue

            # ====== PIPELINE DE NETTOYAGE COMPLET ======
            cleaned_text = clean_text_pipeline(text)

            # Supprimer les stopwords après le nettoyage général
            cleaned_text = remove_stopwords_spacy(cleaned_text)

            if not cleaned_text.strip():
                skipped_count += 1
                continue

            # Sauvegarder le fichier nettoyé
            cleaned_file = os.path.join(config.CLEANED_DIR, os.path.basename(extracted_path))
            with open(cleaned_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            # Mettre à jour le state
            doc["cleaned_text"] = cleaned_text
            doc["cleaned_text_path"] = cleaned_file
            doc["raw_path"] = doc.get("origin_path")
            doc["is_cleaned"] = True
            cleaned_count += 1

        except Exception as e:
            print(f"[Agent1-Cleaning] ❌ Error on {doc.get('doc_id')}: {e}")
            skipped_count += 1

    # Synchroniser le trace_index.csv
    trace_csv_path = config.TRACE_CSV
    if raw_docs:
        fieldnames = ["doc_id", "raw_path", "cleaned_text_path", "is_cleaned"]

        all_keys = set()
        for doc in raw_docs:
            all_keys.update(doc.keys())

        final_fieldnames = fieldnames + [k for k in all_keys if k not in fieldnames and k != "cleaned_text"]

        with open(trace_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(raw_docs)

    print(f"[Agent1-Cleaning] Done: {cleaned_count} cleaned, {skipped_count} skipped")
    print(f"[Agent1-Cleaning] Saved to {config.CLEANED_DIR}")
    print(f"[Agent1-Cleaning] Updated {trace_csv_path}")

    return state