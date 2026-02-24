import os
import json
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from core.config import agent2_config as config


def detect_structure(text):
    if "def " in text or "class " in text:
        return "code"
    if ":" in text and "\n" in text:
        return "dialogue"
    if len(text.split("\n")) > 5:
        return "article"
    return "unknown"


def lexical_diversity(text):
    words = text.split()
    if not words:
        return 0
    return len(set(words)) / len(words)


def run_agent2(state):

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    raw_docs = state.get("raw_docs", [])
    extracted_files = state.get("extracted_files", [])

    if not extracted_files:
        print("[Agent2] No documents found.")
        return state

    print(f"[Agent2] Processing {len(extracted_files)} documents...")

    texts = []
    doc_ids = []
    lengths = []
    structures = []
    diversities = []

    for doc in raw_docs:
        path = doc["extracted_path"]
        doc_id = doc["doc_id"]

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        texts.append(text)
        doc_ids.append(doc_id)
        lengths.append(len(text.split()))
        structures.append(detect_structure(text))
        diversities.append(lexical_diversity(text))

    # ---------- BASIC PROFILE ----------
    total_words = sum(lengths)
    avg_words = int(total_words / len(texts))
    vocab = set(" ".join(texts).split())

    structure_distribution = dict(Counter(structures))

    dataset_profile = {
        "num_docs": len(texts),
        "total_words": total_words,
        "avg_words_per_doc": avg_words,
        "vocab_size": len(vocab),
        "structure_distribution": structure_distribution,
        "avg_lexical_diversity": round(float(np.mean(diversities)), 4)
    }

    with open(config.DATASET_PROFILE, "w", encoding="utf-8") as f:
        json.dump(dataset_profile, f, indent=2, ensure_ascii=False)

    print("[Agent2] Dataset profile saved.")

    # ---------- EMBEDDINGS ----------
    model = SentenceTransformer(config.MODEL_NAME)
    embeddings = model.encode(texts, batch_size=32)
    embeddings = normalize(embeddings)

    kmeans = KMeans(n_clusters=config.NUM_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for label, doc_id in zip(labels, doc_ids):
        label = int(label)
        clusters.setdefault(label, []).append(doc_id)

    with open(config.CLUSTER_REPORT, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    print("[Agent2] Clusters saved.")

    # ---------- ANOMALIES ----------
    anomalies = {
        k: v for k, v in clusters.items() if len(v) <= 2
    }

    # ---------- DATASET TYPE PREDICTION ----------
    if structure_distribution.get("dialogue", 0) > len(texts) * 0.4:
        dataset_type = "chat_dataset"
    elif structure_distribution.get("code", 0) > len(texts) * 0.3:
        dataset_type = "technical_dataset"
    elif structure_distribution.get("article", 0) > len(texts) * 0.5:
        dataset_type = "encyclopedic_articles"
    else:
        dataset_type = "mixed_dataset"

    intelligent_summary = {
        "probable_dataset_type": dataset_type,
        "cluster_sizes": {str(k): len(v) for k, v in clusters.items()},
        "anomalies": anomalies
    }

    with open(config.TOPIC_REPORT, "w", encoding="utf-8") as f:
        json.dump(intelligent_summary, f, indent=2, ensure_ascii=False)

    print("[Agent2] Intelligent analysis saved.")

    return {
        **state,
        "dataset_profile": dataset_profile,
        "clusters": clusters,
        "intelligent_summary": intelligent_summary
    }