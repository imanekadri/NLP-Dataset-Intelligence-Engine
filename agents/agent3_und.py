import os
import re
import json
from collections import Counter

import numpy as np
import spacy
import yake
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from core.config import agent3_config as config
from core.utils import (
    detect_language,
    extract_entities,
    detect_pii,
    analyze_sentiment,
    normalize_text_senior,
    normalize_text_for_embedding,
)


# =========================
# Keyword cleaning
# =========================
def clean_keywords(kws):
    clean = []
    for w in kws:
        w = w.lower().strip()
        if w and w not in ENGLISH_STOP_WORDS and len(w) > 2:
            clean.append(w)
    return clean


# =========================
# Paragraph splitting
# =========================
def split_paragraphs(text, min_len=None):
    """Split on blank lines, then merge fragments shorter than min_len."""
    if min_len is None:
        min_len = config.PARAGRAPH_MIN_LEN

    raw_blocks = re.split(r"\n\s*\n+", text)
    blocks = [re.sub(r"\s+", " ", b).strip() for b in raw_blocks]
    blocks = [b for b in blocks if b]

    paragraphs = []
    buffer = ""
    for b in blocks:
        buffer = (buffer + " " + b).strip() if buffer else b
        if len(buffer) >= min_len:
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        if paragraphs:
            paragraphs[-1] = (paragraphs[-1] + " " + buffer).strip()
        else:
            paragraphs.append(buffer)
    return paragraphs


# =========================
# Subdomain detection (BERTopic on keywords)
# =========================
def detect_subdomains_from_keywords(docs, embedding_model=None):
    keyword_texts = []
    valid_docs = []

    for doc in docs:
        kws = doc.get("key_words", [])
        if isinstance(kws, str):
            kws = kws.split(",")
        kws = clean_keywords(kws)
        if not kws:
            continue
        keyword_texts.append(" ".join(kws))
        valid_docs.append(doc)

    if not keyword_texts:
        return [(doc, "general") for doc in docs]

    if len(keyword_texts) < config.MIN_DOCS_FOR_SUBDOMAINS:
        return [(doc, "general") for doc in valid_docs]

    try:
        topic_model = BERTopic(
            embedding_model=embedding_model or config.EMBEDDING_MODEL,
            min_topic_size=config.MIN_SUBDOMAIN_SIZE,
            nr_topics="auto",
        )
        topics, _ = topic_model.fit_transform(keyword_texts)

        subdomain_names = {}
        for topic_id in set(topics):
            if topic_id == -1:
                continue
            words = topic_model.get_topic(topic_id)
            if not words:
                continue
            words = [w for w, _ in words if w not in ENGLISH_STOP_WORDS]
            subdomain_names[topic_id] = "_".join(words[:2]) if words else "general"
    except Exception as e:
        print(f"[Agent3] WARNING: BERTopic subdomain detection failed, falling back to 'general'. Error: {e}")
        return [(doc, "general") for doc in valid_docs]

    return [(doc, subdomain_names.get(tid, "general")) for doc, tid in zip(valid_docs, topics)]


# =========================
# Intra-document subtopic clustering (transitive, union-find)
# =========================
def cluster_paragraphs_by_similarity(p_vectors, threshold):
    n = len(p_vectors)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    sim = cosine_similarity(np.asarray(p_vectors))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] > threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def slugify(raw, max_len=30):
    slug = re.sub(r"[^\w]+", "_", raw.lower()).strip("_")
    return (slug or "section")[:max_len]


def name_subtopic(cluster_indices, doc_paragraphs, kw_extractor):
    cluster_text = " ".join(doc_paragraphs[idx]["text"] for idx in cluster_indices)
    cluster_entities = []
    for idx in cluster_indices:
        cluster_entities.extend(e["text"] for e in doc_paragraphs[idx]["entities"])

    if cluster_entities:
        raw = "_".join(cluster_entities[:2])
    else:
        kws = kw_extractor.extract_keywords(cluster_text)
        raw = "_".join(kw for kw, _ in kws[:2]) if kws else "general_snippet"
    return slugify(raw)


# =========================
# Helpers
# =========================
def load_models():
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    ner_en = spacy.load(config.SPACY_EN)
    ner_fr = spacy.load(config.SPACY_FR)
    sentiment_model = pipeline(
        "sentiment-analysis",
        model=config.SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
    )
    pii_engine = AnalyzerEngine()
    kw_extractor = yake.KeywordExtractor(lan="multilingual", n=2, dedupLim=0.9, top=5)
    return embedding_model, ner_en, ner_fr, sentiment_model, pii_engine, kw_extractor


def build_paragraphs(
    paragraphs_raw,
    embedding_model,
    sentiment_model,
    pii_engine,
    ner_en,
    ner_fr,
    lang_code,
):
    """Compute embeddings (batched), sentiment, entities, and PII per paragraph."""
    if not paragraphs_raw:
        return [], {}, []

    # Normalize per paragraph (after split, so \n\n boundaries are preserved)
    paragraphs_raw = [normalize_text_senior(p) for p in paragraphs_raw]
    paragraphs_raw = [p for p in paragraphs_raw if p]

    normalized = [normalize_text_for_embedding(p) for p in paragraphs_raw]
    vectors = embedding_model.encode(
        normalized,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
    )

    doc_paragraphs = []
    doc_embeddings = {}
    p_vectors = []

    for idx, p_text in enumerate(paragraphs_raw):
        v = vectors[idx]
        v_list = v.tolist() if hasattr(v, "tolist") else list(v)
        doc_embeddings[f"paragraph_{idx}"] = v_list
        p_vectors.append(v_list)

        sent = analyze_sentiment(sentiment_model, p_text, config.SENTIMENT_TRIGGER_KEYWORDS)

        p_entities = extract_entities(
            p_text, lang_code, ner_en, ner_fr,
            valid_labels=config.VALID_ENTITY_LABELS,
            tech_keywords=config.TECH_KEYWORDS,
            generic_blacklist=config.GENERIC_WORDS_BLACKLIST,
            org_verbs=config.ORG_VERB_SIGNATURES,
        )

        try:
            p_pii = detect_pii(
                pii_engine, p_text, lang_code,
                config.PII_SCORE_THRESHOLD,
                important_pii=config.IMPORTANT_PII,
                tech_keywords=config.TECH_KEYWORDS,
            )
        except Exception:
            p_pii = []

        doc_paragraphs.append({
            "paragraph_id": idx,
            "text": p_text,
            "sentiment": sent,
            "entities": p_entities,
            "pii": p_pii,
        })

    return doc_paragraphs, doc_embeddings, p_vectors


def compute_subtopics(p_vectors, doc_paragraphs, kw_extractor):
    if len(p_vectors) <= 1:
        return [{"subtopic_id": "S1", "name": "document_content", "paragraph_ids": [0]}]

    clusters = cluster_paragraphs_by_similarity(p_vectors, config.SUBTOPIC_SIMILARITY_THRESHOLD)
    subtopics = []
    for sid, cluster_indices in enumerate(clusters, start=1):
        subtopics.append({
            "subtopic_id": f"S{sid}",
            "name": name_subtopic(cluster_indices, doc_paragraphs, kw_extractor),
            "paragraph_ids": cluster_indices,
        })
    return subtopics


def aggregate_globals(doc_paragraphs):
    global_entities, seen_entities = [], set()
    global_pii, seen_pii = [], set()
    for p in doc_paragraphs:
        for ent in p["entities"]:
            if ent["text"] not in seen_entities:
                global_entities.append(ent)
                seen_entities.add(ent["text"])
        for pii in p["pii"]:
            if pii["text"] not in seen_pii:
                global_pii.append(pii)
                seen_pii.add(pii["text"])
    return global_entities, global_pii


def dominant_sentiment(doc_paragraphs):
    sentiments = [p["sentiment"] for p in doc_paragraphs]
    return Counter(sentiments).most_common(1)[0][0] if sentiments else "NEUTRAL"


def load_doc_text(doc):
    text = doc.get("cleaned_text")
    file_path = doc.get("cleaned_text_path", doc.get("extracted_path", ""))
    if not text and file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    return text, file_path


# =========================
# MAIN
# =========================
def run_agent3(state):
    if "raw_docs" not in state:
        print("[Agent3] Missing raw_docs [ERROR]")
        return state

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    categories_dir = os.path.join(config.OUTPUT_DIR, "categories")
    os.makedirs(categories_dir, exist_ok=True)

    embedding_model, ner_en, ner_fr, sentiment_model, pii_engine, kw_extractor = load_models()

    global_report = {}

    topics_groups = {}
    for doc in state["raw_docs"]:
        topics_groups.setdefault(doc.get("topic", "general"), []).append(doc)

    for topic, docs in topics_groups.items():
        sub_results = detect_subdomains_from_keywords(docs, embedding_model=embedding_model)

        for doc, global_subdomain in sub_results:
            text, file_path = load_doc_text(doc)
            if not text:
                continue

            language = detect_language(text)
            lang_code = "fr" if language.startswith("fr") else "en"
            filename = os.path.basename(file_path) if file_path else doc.get("doc_id", "unknown")

            # Split on raw text to preserve paragraph boundaries (\n\n),
            # then normalize each paragraph individually in build_paragraphs.
            paragraphs_raw = split_paragraphs(text)
            doc_paragraphs, doc_embeddings, p_vectors = build_paragraphs(
                paragraphs_raw, embedding_model, sentiment_model,
                pii_engine, ner_en, ner_fr, lang_code,
            )

            subtopics = compute_subtopics(p_vectors, doc_paragraphs, kw_extractor)
            global_entities, global_pii = aggregate_globals(doc_paragraphs)

            sub_path = os.path.join(categories_dir, topic, global_subdomain)
            os.makedirs(sub_path, exist_ok=True)
            json_path = os.path.join(sub_path, f"{os.path.splitext(filename)[0]}.json")

            doc_json = {
                "doc_id": filename,
                "paragraphs": doc_paragraphs,
                "embeddings": doc_embeddings,
                "subtopics": subtopics,
                "global_entities": global_entities,
                "global_pii": global_pii,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc_json, f, indent=2, ensure_ascii=False)

            global_report.setdefault(topic, {}).setdefault(global_subdomain, []).append({
                "doc": filename,
                "path": json_path,
                "sentiment": dominant_sentiment(doc_paragraphs),
            })

    report_path = os.path.join(config.OUTPUT_DIR, "agent3_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(global_report, f, indent=2, ensure_ascii=False)

    print("[Agent3] Production extraction complete [OK]")
    state["agent3_results"] = report_path
    return state
