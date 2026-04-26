import os
import json
import shutil
import numpy as np
import csv
from collections import Counter
from bertopic import BERTopic
from langdetect import detect

from core.config import agent2_config as config
from core.utils import detect_structure, calculate_noise_ratio


def run_agent2(state):

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    raw_docs = state.get("raw_docs", [])

    if not raw_docs:
        print("[Agent2] No documents from Agent1.")
        return state

    # =========================
    # Prepare Keywords
    # =========================

    keywords_texts = []
    doc_metadata = []

    for doc in raw_docs:

        kws = doc.get("key_words", [])

        if not kws:
            continue

        keywords_texts.append(" ".join(kws))
        doc_metadata.append(doc)

    print("[Agent2] Extracting topics from keywords...")

    # =========================
    # Topic Modeling
    # =========================

    topic_model = BERTopic(
        embedding_model=config.MODEL_NAME,
        min_topic_size=config.MIN_TOPIC_SIZE,
        nr_topics="auto",
        calculate_probabilities=True
    )

    topics, _ = topic_model.fit_transform(keywords_texts)

    # =========================
    # Extract Topic Names
    # =========================

    topic_names = {}

    for topic_id in set(topics):

        if topic_id == -1:
            continue

        words = topic_model.get_topic(topic_id)

        if not words:
            continue

        topic_name = "_".join([w for w, _ in words[:2]])

        topic_names[topic_id] = topic_name

    # =========================
    # Create Topic Folders
    # =========================

    categories_dir = config.CATEGORIES_DIR

    if os.path.exists(categories_dir):
        shutil.rmtree(categories_dir)

    os.makedirs(categories_dir, exist_ok=True)

    clusters = {}

    for i, topic_id in enumerate(topics):

        topic_name = topic_names.get(topic_id, "unknown_topic")

        topic_folder = os.path.join(categories_dir, topic_name)

        os.makedirs(topic_folder, exist_ok=True)

        doc_info = doc_metadata[i]

        # Prioritize cleaned_text_path as requested by user
        src_file = doc_info.get("cleaned_text_path", doc_info.get("cleaned_path", doc_info["extracted_path"]))
        dst_file = os.path.join(topic_folder, os.path.basename(src_file))

        shutil.copy2(src_file, dst_file)

        doc_info["topic"] = topic_name

        clusters.setdefault(topic_name, []).append(doc_info["doc_id"])

    # =========================
    # Analyze Dataset
    # =========================

    texts = []
    languages = []
    noise_scores = []
    dialogue_count = 0
    code_count = 0

    for doc in doc_metadata:

        try:
            with open(doc["extracted_path"], "r", encoding="utf-8") as f:
                text = f.read()

            texts.append(text)

            noise_scores.append(calculate_noise_ratio(text))

            dialogue, code = detect_structure(text)

            dialogue_count += int(dialogue)
            code_count += int(code)

            try:
                lang = detect(text[:2000])
            except:
                lang = "unknown"

            languages.append(lang)

        except:
            continue

    avg_noise = float(np.mean(noise_scores)) if noise_scores else 0

    dialogue_ratio = dialogue_count / len(texts) if texts else 0
    code_ratio = code_count / len(texts) if texts else 0

    language_distribution = dict(Counter(languages))

    # =========================
    # Dataset Type
    # =========================

    if dialogue_ratio > 0.4:
        dataset_type = "chat_dataset"
    elif code_ratio > 0.3:
        dataset_type = "technical_dataset"
    elif len(language_distribution) > 2:
        dataset_type = "multilingual_dataset"
    else:
        dataset_type = "mixed_dataset"

    # =========================
    # Dataset Profile
    # =========================

    dataset_profile = {
        "probable_dataset_type": dataset_type,
        "total_documents": len(doc_metadata),
        "topics_detected": list(clusters.keys()),
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "avg_noise_ratio": round(avg_noise, 4),
        "dialogue_ratio": round(dialogue_ratio, 4),
        "code_ratio": round(code_ratio, 4),
        "language_distribution": language_distribution
    }

    with open(config.DATASET_PROFILE, "w", encoding="utf-8") as f:
        json.dump(dataset_profile, f, indent=2, ensure_ascii=False)

    # =========================
    # Topic Report
    # =========================

    topic_report = {
        "dataset_type": dataset_profile["probable_dataset_type"],
        "topics": clusters
    }

    with open(config.TOPIC_REPORT, "w", encoding="utf-8") as f:
        json.dump(topic_report, f, indent=2, ensure_ascii=False)

    # =========================
    # Save Trace CSV
    # =========================
    trace_csv_path = os.path.join(config.OUTPUT_DIR, "trace_index.csv")
    

    
    for doc in doc_metadata:
        if isinstance(doc.get("key_words"), list):
            doc["key_words"] = ", ".join(doc["key_words"])

    if doc_metadata:

        fieldnames = []

        for doc in doc_metadata:
            for key in doc.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(trace_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(doc_metadata)

    print(f"[Agent2] Topics created in: {categories_dir}")
    print(f"[Agent2] Trace saved to: {trace_csv_path}")

    return {
        **state,
        "dataset_profile": dataset_profile,
        "clusters": clusters,
        "categories": categories_dir,
        "raw_docs": doc_metadata
    }