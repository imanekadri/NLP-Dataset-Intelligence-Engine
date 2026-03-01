from transformers import pipeline

class ZeroShotClassifier:

    def __init__(self):
        self.model = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli"
                              ,framework="pt"
        )

    def classify(self, text, labels):
        result = self.model(
            text,
      labels,
            hypothesis_template="Ce message client concerne {}.")
        return {
            "label": result["labels"][0],
            "confidence": result["scores"][0]
        }

