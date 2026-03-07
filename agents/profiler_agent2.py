import os
import json
import shutil
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from langdetect import detect

from core.config import agent2_config as config
from core.utils import detect_structure, calculate_noise_ratio

def run_agent2(state):

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    extracted_files = state.get("extracted_files", [])
    raw_docs = state.get("raw_docs", [])

    if not extracted_files:
        print("[Agent2] No extracted files found from Agent1.")
        return state

    texts = []
    doc_ids = []
    doc_metadata = []  

    for i, file_path in enumerate(extracted_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            texts.append(text)

             ###
            doc_id = None
            for doc in raw_docs:
                if doc.get("extracted_path") == file_path:
                    doc_id = doc.get("doc_id")
                    doc_metadata.append(doc)
                    break

            if not doc_id:
                doc_id = f"doc_{i}"
                doc_metadata.append({"doc_id": doc_id, "extracted_path": file_path})

            doc_ids.append(doc_id)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print("[Agent2] Generating embeddings...")

    model = SentenceTransformer(config.MODEL_NAME)
    embeddings = model.encode(texts, batch_size=config.EMBEDDING_BATCH_SIZE)

    # ----------------- Clustering -----------------
    kmeans = KMeans(
        n_clusters=min(config.NUM_CLUSTERS, len(texts)),
        random_state=config.RANDOM_STATE
    )
    labels = kmeans.fit_predict(embeddings)
    sil_score = silhouette_score(embeddings, labels)

    clusters = {}
    for label, doc_id in zip(labels, doc_ids):
        clusters.setdefault(str(label), []).append(doc_id)

    with open(config.CLUSTER_REPORT, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    # --- BERTopic ---
    print("[Agent2] Extracting topics with BERTopic...")
    topic_model = BERTopic(language="multilingual")
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    cluster_topics = {}
    for row in topic_info.itertuples():
        if row.Topic != -1:
            cluster_topics[str(row.Topic)] = [word for word, _ in topic_model.get_topic(row.Topic)[:5]]

    # --- Organize Physical Categories ---
    categories_dir = config.CATEGORIES_DIR
    if os.path.exists(categories_dir):
        shutil.rmtree(categories_dir)
    os.makedirs(categories_dir, exist_ok=True)

    for i, label in enumerate(labels):
        cluster_id = str(label)
        cluster_folder = os.path.join(categories_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)

        doc_info = doc_metadata[i]
        shutil.copy2(
            doc_info["extracted_path"],
            os.path.join(cluster_folder, os.path.basename(doc_info["extracted_path"]))
        )

    # ----------------- Outliers -----------------
    distances = kmeans.transform(embeddings)
    min_distances = np.min(distances, axis=1)
    threshold = np.percentile(min_distances, 95)
    outliers = [doc_ids[i] for i, d in enumerate(min_distances) if d > threshold]

    # ----------------- Noise + Structure -----------------
    noise_scores = []
    dialogue_count = 0
    code_count = 0
    languages = []

    for text in texts:
        noise_scores.append(calculate_noise_ratio(text))
        dialogue, code = detect_structure(text)
        dialogue_count += int(dialogue)
        code_count += int(code)
        try:
            lang = detect(text[:2000])
        except:
            lang = "unknown"
        languages.append(lang)

    avg_noise = float(np.mean(noise_scores))
    dialogue_ratio = dialogue_count / len(texts)
    code_ratio = code_count / len(texts)
    language_distribution = dict(Counter(languages))

    # ----------------- Dataset Type -----------------
    if dialogue_ratio > 0.4:
        dataset_type = "chat_dataset"
    elif code_ratio > 0.3:
        dataset_type = "technical_dataset"
    elif len(language_distribution) > 2:
        dataset_type = "multilingual_dataset"
    else:
        dataset_type = "mixed_dataset"

    # ----------------- Profile -----------------
    dataset_profile = {
        "probable_dataset_type": dataset_type,
        "total_documents": len(texts),
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "clustering_quality_score": round(float(sil_score), 4),
        "avg_noise_ratio": round(avg_noise, 4),
        "dialogue_ratio": round(dialogue_ratio, 4),
        "code_ratio": round(code_ratio, 4),
        "language_distribution": language_distribution,
        "anomalies": {
            "embedding_outliers": outliers
        }
    }

    with open(config.DATASET_PROFILE, "w", encoding="utf-8") as f:
        json.dump(dataset_profile, f, indent=2, ensure_ascii=False)

    intelligent_summary = {
        "probable_dataset_type": dataset_profile["probable_dataset_type"],
        "cluster_sizes": dataset_profile["cluster_sizes"],
        "topics_per_cluster": cluster_topics,
        "anomalies": dataset_profile["anomalies"]
    }

    with open(config.TOPIC_REPORT, "w", encoding="utf-8") as f:
        json.dump(intelligent_summary, f, indent=2, ensure_ascii=False)

    print(f"[Agent2] Analysis complete. Physical files organized in {categories_dir}")

    return {
        **state,
        "dataset_profile": dataset_profile,
        "clusters": clusters,
        "categories": categories_dir
    }