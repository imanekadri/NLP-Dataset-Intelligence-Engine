class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

    def analyze(self, text: str):
        result = self.model(text)[0]
        label = result["label"]
        score = float(result["score"])

        if "1" in label or "2" in label:
            sentiment = "negative"
        elif "3" in label:
            sentiment = "neutral"
        else:
            sentiment = "positive"

        return {
            "label": sentiment,
            "score": score
        }