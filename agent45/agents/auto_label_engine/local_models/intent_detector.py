from sentence_transformers import SentenceTransformer
import numpy as np

class IntentDetector:

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.intent_examples = {
            "cancel_order": "I want to cancel my order",
            "refund": "I want a refund",
            "complaint": "I am not satisfied"
        }
        self.intent_embeddings = {
            k: self.embedder.encode(v)
            for k, v in self.intent_examples.items()
        }

    def detect(self, text):
        text_embedding = self.embedder.encode(text)
        similarities = {
            intent: np.dot(text_embedding, emb)
            for intent, emb in self.intent_embeddings.items()
        }
        return max(similarities, key=similarities.get)