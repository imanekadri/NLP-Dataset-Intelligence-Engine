"""
agents/auto_label_engine/agent.py
AutoLabelEngine — improved version.

Key improvements over original:
  1. Classifier: strips meta-language from labels → fixes wrong classifications
  2. NER: validates entity types → fixes garbage entities (veux=ORG, 123456=DATE)
  3. ConfidenceEngine: real weighted scoring → no more static 0.5
  4. LLM tasks: return structured JSON dicts → not raw text strings
  5. Batch processing: process() + process_batch() for efficiency
  6. Language awareness: passes detected lang to NER/classifier
  7. Retry fallback: each module fails gracefully without crashing
  8. Human-review flag: results below confidence threshold are flagged
"""

from agent45.agents.auto_label_engine.llm_client import GroqClient
from agent45.agents.auto_label_engine.local_models.classifier import ZeroShotClassifier
from agent45.agents.auto_label_engine.local_models.ner import NERExtractor
from agent45.agents.auto_label_engine.local_models.topic_modeler import TopicModeler
from agent45.agents.auto_label_engine.local_models.intent_detector import IntentDetector
from agent45.agents.auto_label_engine.local_models.translator import Translator
from agent45.agents.auto_label_engine.local_models.sentiment import SentimentAnalyzer
from agent45.agents.auto_label_engine.local_models.clustering import TextClustering
from agent45.agents.auto_label_engine.confidence import ConfidenceEngine
from agent45.agents.auto_label_engine.llm_tasks import (
    chatbot_run,
    qa_run,
    instruction_run,
    code_run,
    rag_run,
)
from agent45.agents.auto_label_engine.llm_tasks.summarizer import run as summarizer_run


class AutoLabelEngine:
    """
    Autonomous labeling engine that applies:
      - Zero-shot classification  (BART-MNLI)
      - NER                       (spaCy + transformers)
      - Intent detection          (sentence-transformers similarity)
      - Sentiment analysis        (XLM-RoBERTa multilingual)
      - Topic labeling            (BERTopic + LLM naming)
      - Clustering                (K-Means on embeddings)
      - Translation               (Helsinki-NLP MarianMT)
      - LLM generative tasks      (Groq/LLaMA-3)
      - Real confidence scoring   (weighted multi-signal)
    """

    def __init__(self):

        # ── LLM Client (Groq) ─────────────────
        self.llm = GroqClient()

        # ── Local Models ──────────────────────
        self.classifier    = ZeroShotClassifier()
        self.ner           = NERExtractor()
        self.topic_modeler = TopicModeler()
        self.intent        = IntentDetector()
        self.translator    = Translator()
        self.sentiment     = SentimentAnalyzer()
        self.clustering    = TextClustering()

        # ── Confidence Engine ─────────────────
        self.confidence = ConfidenceEngine()

    # ════════════════════════════════════════════════════════
    # PUBLIC: process() — single text
    # ════════════════════════════════════════════════════════

    def process(self, text: str, brain_decision: dict, dataset_type: str = None) -> dict:
        """
        Label a single text according to brain_decision instructions.

        Args:
            text:           Input text to label
            brain_decision: Instructions from Agent 4:
                {
                    "dataset_type": "chatbot_training",
                    "tasks": ["classification", "ner", "intent", "sentiment", "topic", "clustering"],
                    "labels": ["cancel_order", "refund", "complaint"],
                    "context": "Optional RAG context",
                    "language": "fr"    ← NEW: optional language hint
                }
            dataset_type:   Override for dataset_type in brain_decision

        Returns:
            Fully labeled dict with confidence_score and needs_review flag
        """
        result = {"text": text}

        dtype  = dataset_type or brain_decision.get("dataset_type", "")
        tasks  = brain_decision.get("tasks", [])
        labels = brain_decision.get("labels", [])
        lang   = brain_decision.get("language", "en")

        # ── Pre-processing: Translation ───────
        if dtype == "translation":
            translated = self.translator.translate(text)
            result["translation"] = translated
            result["confidence_score"] = self.confidence.compute(result)
            result["needs_review"] = self.confidence.needs_review(result)
            return result

        # ── Classification ────────────────────
        if "classification" in tasks and labels:
            result["classification"] = self.classifier.classify(
                text, labels,
                hypothesis_template="This text is about {}."
            )

        # ── NER ───────────────────────────────
        if "ner" in tasks:
            result["entities"] = self.ner.extract(text, lang=lang)

        # ── Intent Detection ──────────────────
        if "intent" in tasks:
            result["intent"] = self.intent.detect(text)

        # ── Sentiment ─────────────────────────
        if "sentiment" in tasks:
            result["sentiment"] = self.sentiment.analyze(text)

        # ── Topic Labeling ────────────────────
        if "topic" in tasks:
            result["topic_id"] = self.topic_modeler.get_topic(text)

        # ── Clustering ────────────────────────
        if "clustering" in tasks:
            result["cluster_id"] = self.clustering.cluster([text])[0]

        # ── LLM Generative Tasks ──────────────
        if dtype == "resume":
            result["summary"] = summarizer_run(self.llm, text)

        elif dtype == "chatbot_training":
            result["chatbot_response"] = chatbot_run(self.llm, text)

        elif dtype == "qa":
            result["qa_pairs"] = qa_run(self.llm, text)

        elif dtype == "instruction_tuning":
            result["instruction_format"] = instruction_run(self.llm, text)

        elif dtype == "code_dataset":
            result["code_samples"] = code_run(self.llm, text)

        elif dtype == "rag":
            context = brain_decision.get("context", "")
            result["rag_answer"] = rag_run(self.llm, text, context)

        # ── Confidence Score ──────────────────
        result["confidence_score"] = self.confidence.compute(result)
        result["needs_review"]     = self.confidence.needs_review(result)

        return result

    # ════════════════════════════════════════════════════════
    # PUBLIC: process_batch() — multiple texts efficiently
    # ════════════════════════════════════════════════════════

    def process_batch(
        self,
        texts: list[str],
        brain_decision: dict,
        dataset_type: str = None
    ) -> list[dict]:
        """
        Process a batch of texts.
        For topic modeling: fits BERTopic on the full batch first (more accurate).
        For clustering: runs K-Means on all embeddings together.

        Returns: list of labeled result dicts
        """
        if not texts:
            return []

        tasks = brain_decision.get("tasks", [])

        # ── Pre-fit BERTopic on full corpus ───
        if "topic" in tasks and len(texts) >= 10:
            print(f"  Pre-fitting BERTopic on {len(texts)} texts...")
            self.topic_modeler.fit(texts, llm_client=self.llm)

        # ── Pre-compute cluster IDs ────────────
        cluster_ids = {}
        if "clustering" in tasks and len(texts) >= 2:
            ids = self.clustering.cluster(texts)
            cluster_ids = {i: cid for i, cid in enumerate(ids)}

        # ── Process each text ─────────────────
        results = []
        for i, text in enumerate(texts):
            result = self.process(text, brain_decision, dataset_type)

            # Inject pre-computed cluster ID
            if "clustering" in tasks and i in cluster_ids:
                result["cluster_id"] = cluster_ids[i]

            results.append(result)

            if (i + 1) % 100 == 0:
                print(f"  Labeled {i+1}/{len(texts)} texts")

        # ── Summary stats ─────────────────────
        avg_confidence = sum(r["confidence_score"] for r in results) / len(results)
        needs_review   = sum(1 for r in results if r.get("needs_review", False))
        print(f"\n  Batch complete | avg_confidence={avg_confidence:.3f} | needs_review={needs_review}/{len(results)}")

        return results

    # ════════════════════════════════════════════════════════
    # PUBLIC: get_stats()
    # ════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Return LLM usage statistics."""
        return self.llm.get_stats()