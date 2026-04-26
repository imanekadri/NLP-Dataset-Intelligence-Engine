import os
import re
import json
import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from core.config import agent3_config as config
from core.utils import detect_language, compute_embedding, extract_entities, detect_pii, analyze_sentiment, normalize_text_senior


# =========================
# التحقق التلقائي من ضرورة تحليل المشاعر
# =========================
def auto_should_skip_sentiment(topic_name, classifier):
    """
    Decides automatically if sentiment analysis should be skipped 
    using zero-shot classification on the topic name.
    """
    if not classifier:
        return False
        
    labels = ["factual/technical/objective", "subjective/opinionated/emotive"]
    try:
        res = classifier(topic_name, candidate_labels=labels)
        best_label = res["labels"][0]
        # Skip if the most likely label is factual
        return best_label == "factual/technical/objective"
    except:
        return False


# =========================
# تنظيف الكلمات
# =========================
def clean_keywords(kws):
    clean = []
    for w in kws:
        w = w.lower().strip()
        if w and w not in ENGLISH_STOP_WORDS and len(w) > 2:
            clean.append(w)
    return clean


# =========================
# تقسيم الفقرات
# =========================
def split_paragraphs(text, min_len=40):
    paragraphs = []
    buffer = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        buffer += " " + line

        if len(buffer) >= min_len:
            paragraphs.append(buffer.strip())
            buffer = ""

    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs


# =========================
# Subdomain Detection من keywords
# =========================
def detect_subdomains_from_keywords(docs):

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

    # Guard: Use a smaller limit for BERTopic to prevent k >= N error
    if len(keyword_texts) < config.MIN_DOCS_FOR_SUBDOMAINS:
        return [(doc, "general") for doc in valid_docs]

    try:
        topic_model = BERTopic(
            embedding_model=config.EMBEDDING_MODEL,
            min_topic_size=config.MIN_SUBDOMAIN_SIZE,
            nr_topics="auto"
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

            name = "_".join(words[:2]) if words else "general"

            subdomain_names[topic_id] = name

    except Exception as e:
        print(f"[Agent3] WARNING: BERTopic subdomain detection failed, falling back to 'general'. Error: {e}")
        return [(doc, "general") for doc in valid_docs]

    results = []

    for doc, topic_id in zip(valid_docs, topics):
        sub = subdomain_names.get(topic_id, "general")
        results.append((doc, sub))

    return results

# =========================
# Sub-topic Detection من keywords
# =========================
def detect_subtopics_from_keywords(docs):
    """Alias for detect_subdomains to align with user naming"""
    return detect_subdomains_from_keywords(docs)


# =========================
# MAIN
# =========================
def run_agent3(state):
    import yake
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if "raw_docs" not in state:
        print("[Agent3] Missing raw_docs ❌")
        return state

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    categories_dir = os.path.join(config.OUTPUT_DIR, "categories")
    os.makedirs(categories_dir, exist_ok=True)

    # 1. Load models
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    ner_en = spacy.load(config.SPACY_EN)
    ner_fr = spacy.load(config.SPACY_FR)
    sentiment_model = pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL, truncation=True, max_length=512)
    pii_engine = AnalyzerEngine()
    
    # Keyword extractor for naming subtopics
    kw_extractor = yake.KeywordExtractor(lan="multilingual", n=2, dedupLim=0.9, top=5)

    global_report = {}
    
    # Group docs by topic
    topics_groups = {}
    for doc in state["raw_docs"]:
        topic = doc.get("topic", "general")
        topics_groups.setdefault(topic, []).append(doc)

    for topic, docs in topics_groups.items():
        # Global topic-level subdomain analysis (BERTopic)
        sub_results = detect_subdomains_from_keywords(docs)

        for doc, global_subdomain in sub_results:
            # Load raw text
            text = doc.get("cleaned_text")
            file_path = doc.get("cleaned_text_path", doc.get("extracted_path", ""))
            if not text and file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            if not text: continue

            # 2. SENIOR NORMALIZATION
            text = normalize_text_senior(text)
            language = detect_language(text)
            lang_code = "fr" if language.startswith("fr") else "en"
            filename = os.path.basename(file_path) if file_path else doc.get("doc_id", "unknown")

            # 3. PARAGRAPH-LEVEL PROCESSING
            paragraphs_raw = split_paragraphs(text)
            doc_paragraphs = []
            doc_embeddings = {}
            p_vectors = [] # For clustering

            for idx, p_text in enumerate(paragraphs_raw):
                # Embedding (Paragraph-level)
                v = compute_embedding(embedding_model, p_text)
                v_list = v.tolist() if hasattr(v, "tolist") else v
                doc_embeddings[f"paragraph_{idx}"] = v_list
                p_vectors.append(v_list)

                # Sentiment (Triggered)
                sent = analyze_sentiment(sentiment_model, p_text, config.SENTIMENT_TRIGGER_KEYWORDS)

                # Local Entities (Object-based)
                p_entities = extract_entities(
                    p_text, lang_code, ner_en, ner_fr, 
                    valid_labels=config.VALID_ENTITY_LABELS,
                    tech_keywords=config.TECH_KEYWORDS,
                    generic_blacklist=config.GENERIC_WORDS_BLACKLIST,
                    org_verbs=config.ORG_VERB_SIGNATURES
                )

                # Local PII (Object-based)
                try:
                    p_pii = detect_pii(
                        pii_engine, p_text, lang_code, 
                        config.PII_SCORE_THRESHOLD, 
                        important_pii=config.IMPORTANT_PII,
                        tech_keywords=config.TECH_KEYWORDS
                    )
                except: p_pii = []

                doc_paragraphs.append({
                    "paragraph_id": idx,
                    "text": p_text,
                    "sentiment": sent,
                    "entities": p_entities,
                    "pii": p_pii
                })

            # 4. SUBTOPIC DETECTION (Intra-document clustering)
            subtopics = []
            if len(p_vectors) > 1:
                # Simple similarity-based grouping
                sim_matrix = cosine_similarity(p_vectors)
                visited = [False] * len(p_vectors)
                sid_counter = 1
                
                for i in range(len(p_vectors)):
                    if visited[i]: continue
                    
                    cluster_indices = [i]
                    visited[i] = True
                    for j in range(i + 1, len(p_vectors)):
                        if not visited[j] and sim_matrix[i][j] > 0.6: # Threshold 0.6
                            cluster_indices.append(j)
                            visited[j] = True
                    
                    # Naming logic: snake_case keywords from cluster
                    cluster_text = " ".join([doc_paragraphs[idx]["text"] for idx in cluster_indices])
                    kws = kw_extractor.extract_keywords(cluster_text)
                    # Priority: Entity words > Keywords
                    cluster_entities = []
                    for idx in cluster_indices:
                        cluster_entities.extend([e["text"] for e in doc_paragraphs[idx]["entities"]])
                    
                    if cluster_entities:
                        raw_name = cluster_entities[0]
                    elif kws:
                        raw_name = kws[0][0]
                    else:
                        raw_name = "general_snippet"
                    
                    name_slug = re.sub(r'[^\w]+', '_', raw_name.lower()).strip('_')
                    if not name_slug: name_slug = "section"

                    subtopics.append({
                        "subtopic_id": f"S{sid_counter}",
                        "name": name_slug[:30], # Cap length
                        "paragraph_ids": cluster_indices
                    })
                    sid_counter += 1
            else:
                # Fallback for single paragraph
                subtopics.append({"subtopic_id": "S1", "name": "document_content", "paragraph_ids": [0]})

            # 5. GLOBAL AGGREGATION
            global_entities = []
            seen_entities = set()
            global_pii = []
            seen_pii = set()
            
            for p in doc_paragraphs:
                for ent in p["entities"]:
                    if ent["text"] not in seen_entities:
                        global_entities.append(ent)
                        seen_entities.add(ent["text"])
                for p_item in p["pii"]:
                    if p_item["text"] not in seen_pii:
                        global_pii.append(p_item)
                        seen_pii.add(p_item["text"])

            # 6. SAVE PRODUCTION JSON
            sub_path = os.path.join(categories_dir, topic, global_subdomain)
            os.makedirs(sub_path, exist_ok=True)
            json_path = os.path.join(sub_path, f"{os.path.splitext(filename)[0]}.json")

            doc_json = {
                "doc_id": filename,
                "paragraphs": doc_paragraphs,
                "embeddings": doc_embeddings,
                "subtopics": subtopics,
                "global_entities": global_entities,
                "global_pii": global_pii
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc_json, f, indent=2, ensure_ascii=False)

            # Report update
            global_report.setdefault(topic, {}).setdefault(global_subdomain, []).append({
                "doc": filename,
                "path": json_path,
                "sentiment": (max(set([p["sentiment"] for p in doc_paragraphs]), key=[p["sentiment"] for p in doc_paragraphs].count) if doc_paragraphs else "NEUTRAL")
            })

    # Save final report
    report_path = os.path.join(config.OUTPUT_DIR, "agent3_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(global_report, f, indent=2, ensure_ascii=False)

    print("[Agent3] Production extraction complete ✅")
    state["agent3_results"] = report_path
    return state