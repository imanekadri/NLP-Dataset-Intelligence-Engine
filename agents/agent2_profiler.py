import os
import json
import random
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from pipeline.state import NLPPipelineState
from pipeline.config import PipelineConfig
from utils.text_analysis import (
    load_texts_from_folder,
    calculate_noise_ratio,
    detect_structure,
    detect_languages,
)


def _sample(lst, n, seed=42):
    """Retourne lst complet si len <= n, sinon un échantillon aléatoire de taille n."""
    if len(lst) <= n:
        return lst
    random.seed(seed)
    return random.sample(lst, n)


def run_profiler(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 2 — Dataset Profiler NLP.

    Stratégie de sampling :
    ┌──────────────────────────┬──────────────────────────────────────────┐
    │ Étape                    │ Corpus utilisé                           │
    ├──────────────────────────┼──────────────────────────────────────────┤
    │ Stats, bruit, structure  │ TOUS les textes (corpus complet)         │
    │ Détection de langue      │ Sample LANG_SAMPLE_SIZE (200 par défaut) │
    │ Embeddings + Clustering  │ Sample MAX_DOCS_FOR_EMBEDDING (10 000)   │
    │ BERTopic                 │ Sample MAX_DOCS_FOR_TOPICS (5 000)       │
    └──────────────────────────┴──────────────────────────────────────────┘
    """
    config = PipelineConfig()
    config.ensure_dirs()

    extracted_dir = config.EXTRACTED_DIR
    print(f"[Agent 2] Loading texts from {extracted_dir}...")
    all_texts = load_texts_from_folder(extracted_dir)

    if not all_texts:
        print("[Agent 2] No texts found, skipping profiler.")
        return {**state, "texts": [], "dataset_profile": {}}

    # Filtrer les textes vides / trop courts
    all_texts = [t for t in all_texts if t and len(t.strip()) > 20]
    total_corpus = len(all_texts)
    print(f"[Agent 2] Corpus total : {total_corpus} texts.")

    # ── Stats sur le corpus complet ──────────────────────────────────────────
    print("[Agent 2] Computing statistics on full corpus...")
    word_counts = [len(t.split()) for t in all_texts]
    avg_length = sum(word_counts) / total_corpus
    # vocab sur un sample pour la mémoire (join de 100K docs = RAM)
    vocab_sample = _sample(all_texts, 20_000, config.RANDOM_STATE)
    all_words = " ".join(vocab_sample).lower().split()
    vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0

    noise_scores = [calculate_noise_ratio(t) for t in all_texts]
    avg_noise = sum(noise_scores) / total_corpus

    structure_type = detect_structure(all_texts[0])
    language_distribution = detect_languages(all_texts, config.LANG_SAMPLE_SIZE)

    # ── Embeddings sur un sample représentatif ───────────────────────────────
    embed_texts = _sample(all_texts, config.MAX_DOCS_FOR_EMBEDDING, config.RANDOM_STATE)
    if len(all_texts) > config.MAX_DOCS_FOR_EMBEDDING:
        print(f"[Agent 2] Embedding sample : {len(embed_texts)}/{total_corpus} docs "
              f"(ajuste MAX_DOCS_FOR_EMBEDDING dans config.py pour plus).")
    else:
        print(f"[Agent 2] Computing embeddings for all {len(embed_texts)} docs...")

    embedder = SentenceTransformer(config.MODEL_DIR)
    embeddings_array = embedder.encode(
        embed_texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
    )

    n_clusters = min(config.NUM_CLUSTERS, len(embed_texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    cluster_distribution = {str(k): int(v) for k, v in Counter(cluster_labels).items()}

    # ── BERTopic sur sous-sample des embeddings déjà calculés ────────────────
    topics_output = {}
    if config.USE_TOPIC_MODELING and len(embed_texts) >= config.MIN_TOPIC_SIZE:
        try:
            from bertopic import BERTopic

            if len(embed_texts) > config.MAX_DOCS_FOR_TOPICS:
                indices = _sample(list(range(len(embed_texts))), config.MAX_DOCS_FOR_TOPICS, config.RANDOM_STATE)
                topic_texts = [embed_texts[i] for i in indices]
                topic_embeddings = embeddings_array[indices]
            else:
                topic_texts = embed_texts
                topic_embeddings = embeddings_array

            print(f"[Agent 2] BERTopic on {len(topic_texts)} docs (pre-computed embeddings, no re-download)...")
            topic_model = BERTopic(min_topic_size=config.MIN_TOPIC_SIZE)
            topics, _ = topic_model.fit_transform(topic_texts, embeddings=topic_embeddings)
            topics_output = {str(k): int(v) for k, v in Counter(topics).items()}
        except Exception as e:
            print(f"[Agent 2] BERTopic failed: {e}")

    # ── Inférence du type de dataset ─────────────────────────────────────────
    dataset_type = "General"
    if structure_type == "Code":
        dataset_type = "Code Dataset"
    elif structure_type == "Dialogue":
        dataset_type = "Chat / Conversation Dataset"
    if avg_noise > config.NOISE_THRESHOLD:
        dataset_type += " (High Noise)"

    # ── Profil ────────────────────────────────────────────────────────────────
    profile = {
        "statistical": {
            "total_docs_corpus": total_corpus,
            "docs_embedded": len(embed_texts),
            "avg_length_words": round(avg_length, 2),
            "vocab_richness": round(vocab_richness, 4),
        },
        "structure": structure_type,
        "language_distribution": language_distribution,
        "noise_level": round(avg_noise, 4),
        "clustering": cluster_distribution,
        "topics": topics_output,
        "dataset_probable": dataset_type,
    }

    with open(os.path.join(config.PROFILES_DIR, "dataset_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    with open(os.path.join(config.PROFILES_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_distribution, f, indent=2, ensure_ascii=False)
    if topics_output:
        with open(os.path.join(config.PROFILES_DIR, "topics.json"), "w", encoding="utf-8") as f:
            json.dump(topics_output, f, indent=2, ensure_ascii=False)

    print(f"[Agent 2] Done — {dataset_type} | corpus: {total_corpus} | embedded: {len(embed_texts)}")

    # Construire le dict d'embeddings (doc_id → vecteur)
    raw_docs = state.get("raw_docs", [])
    embeddings_dict = {}
    for i, emb in enumerate(embeddings_array):
        doc_id = raw_docs[i]["doc_id"] if i < len(raw_docs) else f"text_{i}"
        embeddings_dict[doc_id] = emb.tolist()

    return {
        **state,
        "texts": embed_texts,
        "embeddings": embeddings_dict,
        "topic_clusters": topics_output,
        "dataset_profile": profile,
    }
