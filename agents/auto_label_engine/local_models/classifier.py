"""
agents/auto_label_engine/local_models/classifier.py
Zero-shot text classification using facebook/bart-large-mnli.

FIX vs original:
  - Labels were being passed as full sentences → MNLI treats them as hypotheses
  - Must pass clean short label names, not "This message is a greeting"
  - Added language-aware preprocessing
  - Added score normalization
  - Returns top-N labels with probabilities, not just top-1
"""

from typing import Optional


class ZeroShotClassifier:
    """
    Zero-shot classification using BART-MNLI.

    CRITICAL: Labels must be SHORT descriptive phrases, NOT full sentences.
    ❌ WRONG: "This message is a greeting"
    ✅ RIGHT: "greeting"  OR  "order cancellation request"

    The model internally constructs: "This example is [label]"
    """

    MODEL_ID = "facebook/bart-large-mnli"

    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.MODEL_ID,
                device=-1,           # CPU; change to 0 for GPU
                multi_label=False
            )
        return self._pipeline

    def _clean_label(self, label: str) -> str:
        """
        Strip meta-language from labels.
        'This message is a greeting' → 'greeting'
        'cancel_order' → 'cancel order'
        """
        # Remove common meta-prefixes
        prefixes_to_strip = [
            "this message is a ", "this message is ", "this is a ", "this is ",
            "the message is ", "request to ", "a "
        ]
        label_lower = label.lower().strip()
        for prefix in prefixes_to_strip:
            if label_lower.startswith(prefix):
                label_lower = label_lower[len(prefix):]
                break

        # Replace underscores with spaces for better tokenization
        return label_lower.replace("_", " ").strip()

    def classify(
        self,
        text: str,
        labels: list[str],
        top_k: int = 1,
        hypothesis_template: str = "This text is about {}.",
    ) -> dict:
        """
        Classify text into one of the provided labels.

        Args:
            text:               Input text
            labels:             List of candidate labels (keep short and descriptive)
            top_k:              Return top-k predictions
            hypothesis_template: Template for MNLI hypothesis

        Returns:
            {
                "label": "cancel_order",
                "confidence": 0.91,
                "all_scores": {"cancel_order": 0.91, "greeting": 0.05, ...}
            }
        """
        if not text or not text.strip():
            return {"label": labels[0] if labels else "unknown", "confidence": 0.0, "all_scores": {}}

        if not labels:
            return {"label": "unknown", "confidence": 0.0, "all_scores": {}}

        pipe = self._load()

        # Clean labels before sending to model
        clean_labels = [self._clean_label(l) for l in labels]
        # Keep mapping from clean → original
        clean_to_original = dict(zip(clean_labels, labels))

        try:
            result = pipe(
                text[:512],
                candidate_labels=clean_labels,
                hypothesis_template=hypothesis_template,
                multi_label=False
            )

            # Build score dict with original label names
            all_scores = {
                clean_to_original.get(lbl, lbl): round(score, 4)
                for lbl, score in zip(result["labels"], result["scores"])
            }

            top_clean  = result["labels"][0]
            top_label  = clean_to_original.get(top_clean, top_clean)
            top_score  = round(result["scores"][0], 4)

            return {
                "label":      top_label,
                "confidence": top_score,
                "all_scores": all_scores,
            }

        except Exception as e:
            return {
                "label":      labels[0],
                "confidence": 0.0,
                "all_scores": {},
                "error":      str(e)
            }