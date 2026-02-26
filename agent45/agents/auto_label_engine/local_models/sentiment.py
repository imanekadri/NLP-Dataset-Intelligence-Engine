from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

    def analyze(self, text: str) -> str:
        result = self.model(text)[0]
        label = result["label"]

        if "1" in label or "2" in label:
            return "negative"
        elif "3" in label:
            return "neutral"
        else:
            return "positive"
