"""
agents/auto_label_engine/confidence.py
Real confidence scoring — weighted average of all task scores.

FIX vs original:
  - Original returned static 0.5 for all LLM tasks
  - Now computes a weighted score based on which tasks ran and their scores
  - Penalizes low-quality signals (short text, no entities found, etc.)
  - Flags low-confidence results for human review
"""


REVIEW_THRESHOLD = 0.50   # Below this → flag for human review


class ConfidenceEngine:
    """
    Computes a real confidence score (0.0 → 1.0) for a labeled result.

    Scoring logic:
    - Each completed task contributes a weighted score
    - Tasks with explicit model confidence use that value
    - LLM tasks get a base score of 0.70 (LLMs are generally reliable but not calibrated)
    - Text quality signals (length, language detection) modulate the final score
    - Final score = weighted mean of all available signals
    """

    # Task weights — higher = more important to final confidence
    WEIGHTS = {
        "classification": 0.30,
        "intent":         0.25,
        "sentiment":      0.20,
        "entities":       0.10,
        "topic_id":       0.10,
        "cluster_id":     0.05,
    }

    # Base scores for LLM-generated outputs (not directly calibrated)
    LLM_BASE_SCORES = {
        "summary":              0.72,
        "chatbot_response":     0.70,
        "qa_pairs":             0.70,
        "instruction_format":   0.72,
        "code_samples":         0.75,
        "rag_answer":           0.70,
        "translation":          0.80,   # Translation models are well-calibrated
    }

    def compute(self, result: dict) -> float:
        """
        Compute a real confidence score for the full label result.

        Args:
            result: The output dict from AutoLabelEngine.process()

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        text = result.get("text", "")

        # ── Base text quality score ────────────
        text_quality = self._text_quality(text)

        scores    = []
        weights   = []

        # ── Classification ─────────────────────
        clf = result.get("classification")
        if clf and isinstance(clf, dict):
            score = clf.get("confidence", 0.5)
            scores.append(score)
            weights.append(self.WEIGHTS["classification"])

        # ── Intent ────────────────────────────
        intent = result.get("intent")
        if intent and isinstance(intent, dict):
            score = intent.get("confidence", 0.5)
            # Penalize "unknown" intent
            if intent.get("intent") == "unknown":
                score = min(score, 0.3)
            scores.append(score)
            weights.append(self.WEIGHTS["intent"])

        # ── Sentiment ─────────────────────────
        sentiment = result.get("sentiment")
        if sentiment and isinstance(sentiment, dict):
            score = sentiment.get("score", 0.5)
            scores.append(score)
            weights.append(self.WEIGHTS["sentiment"])

        # ── NER (entities) ────────────────────
        entities = result.get("entities")
        if entities is not None:
            # More entities = higher signal (but cap it)
            ner_score = min(0.5 + len(entities) * 0.1, 0.95)
            scores.append(ner_score)
            weights.append(self.WEIGHTS["entities"])

        # ── Topic ─────────────────────────────
        topic = result.get("topic_id")
        if topic is not None:
            topic_score = 0.75 if isinstance(topic, dict) else 0.6
            if isinstance(topic, dict):
                topic_score = topic.get("confidence", 0.6)
            scores.append(topic_score)
            weights.append(self.WEIGHTS["topic_id"])

        # ── Clustering ────────────────────────
        cluster = result.get("cluster_id")
        if cluster is not None:
            scores.append(0.65)
            weights.append(self.WEIGHTS["cluster_id"])

        # ── LLM generative tasks ──────────────
        for key, base_score in self.LLM_BASE_SCORES.items():
            if key in result and result[key]:
                # Check if the LLM output is substantial
                output = result[key]
                if isinstance(output, str) and len(output.strip()) > 20:
                    scores.append(base_score)
                    weights.append(0.15)
                elif isinstance(output, (dict, list)):
                    scores.append(base_score)
                    weights.append(0.15)

        # ── Compute weighted average ───────────
        if not scores:
            return round(text_quality * 0.4, 4)   # No tasks run = low confidence

        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        base_score   = weighted_sum / total_weight

        # Modulate by text quality
        final_score = base_score * 0.85 + text_quality * 0.15

        return round(min(max(final_score, 0.0), 1.0), 4)

    def _text_quality(self, text: str) -> float:
        """
        Score text quality (0.0 → 1.0).
        Very short, empty, or numeric-only texts get lower scores.
        """
        if not text or not text.strip():
            return 0.0

        words = text.strip().split()
        n = len(words)

        if n == 0:
            return 0.0
        if n == 1:
            return 0.35
        if n <= 3:
            return 0.55
        if n <= 10:
            return 0.75
        if n <= 50:
            return 0.90
        return 0.95

    def needs_review(self, result: dict) -> bool:
        """Returns True if this result should be flagged for human review."""
        score = result.get("confidence_score", 0.0)
        return score < REVIEW_THRESHOLD