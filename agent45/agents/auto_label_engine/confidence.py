class ConfidenceEngine:

    def compute(self, outputs: dict):

        scores = []

        if "classification" in outputs:
            scores.append(outputs["classification"].get("confidence", 0))

        if "sentiment" in outputs:
            scores.append(outputs["sentiment"].get("score", 0))

        if "intent_score" in outputs:
            scores.append(outputs.get("intent_score", 0))

        if not scores:
            return 0.5

        return sum(scores) / len(scores)