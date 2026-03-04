from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class IntentDetector:

    def __init__(self, threshold=0.55):
        """
        threshold: minimum similarity score to accept an intent
        """
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

        # -------------------------------
        # Intent examples (MULTIPLE each)
        # -------------------------------
        self.intent_examples = {

            "cancel_order": [
                "I want to cancel my order",
                "Please cancel order 123",
                "Je veux annuler ma commande",
                "Annuler ma commande",
                "Stop my purchase",
                "Cancel it immediately"
            ],

            "refund_request": [
                "I want a refund",
                "Give me my money back",
                "Je veux un remboursement",
                "Refund my payment",
                "Return my money"
            ],

            "complaint": [
                "I am not satisfied",
                "This product is terrible",
                "Je ne suis pas satisfait",
                "Very bad service",
                "I am unhappy with this"
            ],

            "greeting": [
                "Hello",
                "Hi there",
                "Bonjour",
                "Good morning",
                "Hey"
            ],

            "order_status": [
                "Where is my order?",
                "Track my order",
                "Has my package been shipped?",
                "Où est ma commande ?",
                "Is my order delivered?"
            ]
        }

        # Precompute embeddings once
        self.intent_embeddings = {
            intent: self.embedder.encode(sentences)
            for intent, sentences in self.intent_examples.items()
        }

    # ----------------------------------
    # Detect intent
    # ----------------------------------
    def detect(self, text):

        if not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0
            }

        text_embedding = self.embedder.encode([text])

        similarities = {}

        for intent, embeddings in self.intent_embeddings.items():
            sims = cosine_similarity(text_embedding, embeddings)
            similarities[intent] = float(np.max(sims))

        # Get best intent
        best_intent = max(similarities, key=similarities.get)
        confidence = similarities[best_intent]

        # Apply threshold
        if confidence < self.threshold:
            return {
                "intent": "unknown",
                "confidence": confidence
            }

        return {
            "intent": best_intent,
            "confidence": round(confidence, 4)
        }