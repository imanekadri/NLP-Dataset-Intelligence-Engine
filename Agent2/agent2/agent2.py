# agent2.py
import os
import json
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from bertopic import BERTopic

from config import Agent2Config
from utils import (
    load_texts_from_folder,
    calculate_noise_ratio,
    detect_structure,
    detect_languages
)


class DatasetProfilerNLP:
    def __init__(self):
        # Load configuration
        self.cfg = Agent2Config()
        # Load embedding model
        self.embedder = SentenceTransformer("local_model")

        # Load BERTopic model if needed
        if self.cfg.USE_TOPIC_MODELING:
            self.topic_model = BERTopic()

    def run(self, state: dict):
        #  Extracted path
        extracted_path = state.get("extracted_path", self.cfg.INPUT_FOLDER)
        if not extracted_path or not os.path.exists(extracted_path):
            raise ValueError(f"Extracted path not found: {extracted_path}")

        #  Load texts from utils
        texts = load_texts_from_folder(extracted_path)
        if not texts:
            raise ValueError("No texts found in extracted folder.")

        #  Statistical Analysis
        total_docs = len(texts)
        word_counts = [len(t.split()) for t in texts]
        avg_length = sum(word_counts) / total_docs
        all_words = " ".join(texts).lower().split()
        vocab_richness = len(set(all_words)) / len(all_words)

        #  Noise Analysis
        noise_scores = [calculate_noise_ratio(t) for t in texts]
        avg_noise = sum(noise_scores) / total_docs

        #  Structure Detection
        structure_type = detect_structure(texts[0])

        #  Language Detection
        language_distribution = detect_languages(texts, self.cfg.LANG_SAMPLE_SIZE)

        #  Embeddings + Clustering
        embeddings = self.embedder.encode(texts)
        kmeans = KMeans(
            n_clusters=min(self.cfg.NUM_CLUSTERS, total_docs),
            random_state=self.cfg.RANDOM_STATE,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_distribution = dict(Counter(cluster_labels))

        #  Topic Modeling
        topics_output = {}
        if self.cfg.USE_TOPIC_MODELING:
            topics, _ = self.topic_model.fit_transform(texts)
            topics_output = dict(Counter(topics))

        #  Dataset Type Inference
        dataset_type = "General"
        if structure_type == "Code":
            dataset_type = "Code Dataset"
        elif structure_type == "Dialogue":
            dataset_type = "Chat / Conversation Dataset"
        if avg_noise > self.cfg.NOISE_THRESHOLD:
            dataset_type += " (High Noise)"

        #  Build profile
        profile = {
            "statistical": {
                "total_docs": total_docs,
                "avg_length_words": round(avg_length, 2),
                "vocab_richness": round(vocab_richness, 4)
            },
            "structure": structure_type,
            "language_distribution": language_distribution,
            "noise_level": round(avg_noise, 4),
            "clustering": cluster_distribution,
            "topics": topics_output,
            "dataset_probable": dataset_type
        }

        # Save profile in state
        state["dataset_profile"] = profile

        #  Save dataset_profile.json
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        with open(self.cfg.DATASET_PROFILE, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        #  Save clusters.json
        with open(self.cfg.CLUSTER_REPORT, "w", encoding="utf-8") as f:
            json.dump(cluster_distribution, f, indent=2, ensure_ascii=False)

        #  Save topics.json
        if self.cfg.USE_TOPIC_MODELING:
            with open(self.cfg.TOPIC_REPORT, "w", encoding="utf-8") as f:
                json.dump(topics_output, f, indent=2, ensure_ascii=False)

        return state


# ==============================
# Run directly
# ==============================
if __name__ == "__main__":
    profiler = DatasetProfilerNLP()
    state = {"extracted_path": r"E:\Documents\MY PROJECT AI\NLP\agent1\output\extracted",}
    state = profiler.run(state)
    