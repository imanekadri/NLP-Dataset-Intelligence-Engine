from .llm_client import GrokClient

# Local NLP models
from .local_models.classifier import ZeroShotClassifier
from .local_models.ner import NERExtractor
from .local_models.topic_modeler import TopicModeler
from .local_models.intent_detector import IntentDetector
from .local_models.translator import Translator
from .local_models.sentiment import SentimentAnalyzer
from .local_models.clustering import TextClustering

# Confidence engine
from .confidence import ConfidenceEngine

# LLM Tasks
from .llm_tasks import (
    summarizer,
    chatbot,
    qa_generator,
    instruction_builder,
    code_dataset,
    rag_generator
)


class AutoLabelEngine:

    def __init__(self):

        # =============================
        # 🔹 LLM CLIENT (Groq)
        # =============================
        self.llm = GrokClient()

        # =============================
        # 🔹 LOCAL MODELS
        # =============================
        self.classifier = ZeroShotClassifier()
        self.ner = NERExtractor()
        self.topic_modeler = TopicModeler()
        self.intent_detector = IntentDetector()
        self.translator = Translator()
        self.sentiment = SentimentAnalyzer()
        self.clustering = TextClustering()

        # =============================
        # 🔹 CONFIDENCE ENGINE
        # =============================
        self.confidence = ConfidenceEngine()

    def process(self, text: str, brain_decision: dict):

        """
        brain_decision example:

        {
            "dataset_type": "chatbot_training",
            "tasks": [
                "classification",
                "ner",
                "intent",
                "sentiment",
                "topic",
                "clustering"
            ],
            "labels": ["cancel_order", "refund", "complaint"],
            "context": "Optional RAG context"
        }
        """

        result = {"text": text}

        dataset_type = brain_decision.get("dataset_type")
        tasks = brain_decision.get("tasks", [])

        # =====================================================
        # 🔹 TRANSLATION (Pre-processing step if required)
        # =====================================================
        if dataset_type == "translation":
            result["translation"] = self.translator.translate(text)

        # =====================================================
        # 🔹 CLASSIFICATION
        # =====================================================
        if "classification" in tasks:
            labels = brain_decision.get("labels", [])
            if labels:
                result["classification"] = self.classifier.classify(text, labels)

        # =====================================================
        # 🔹 NER
        # =====================================================
        if "ner" in tasks:
            result["entities"] = self.ner.extract(text)

        # =====================================================
        # 🔹 INTENT DETECTION
        # =====================================================
        if "intent" in tasks:
            result["intent"] = self.intent_detector.detect(text)

        # =====================================================
        # 🔹 SENTIMENT
        # =====================================================
        if "sentiment" in tasks:
            result["sentiment"] = self.sentiment.analyze(text)

        # =====================================================
        # 🔹 TOPIC LABELING
        # =====================================================
        if "topic" in tasks:
            result["topic_id"] = self.topic_modeler.get_topic(text)

        # =====================================================
        # 🔹 CLUSTERING (usually batch-based)
        # =====================================================
        if "clustering" in tasks:
            result["cluster_id"] = self.clustering.cluster([text])[0]

        # =====================================================
        # 🔹 LLM GENERATIVE TASKS
        # =====================================================

        if dataset_type == "resume":
            result["summary"] = summarizer.run(self.llm, text)

        if dataset_type == "chatbot_training":
            result["chatbot_response"] = chatbot.run(self.llm, text)

        if dataset_type == "qa":
            result["qa_pairs"] = qa_generator.run(self.llm, text)

        if dataset_type == "instruction_tuning":
            result["instruction_format"] = instruction_builder.run(self.llm, text)

        if dataset_type == "code_dataset":
            result["code_samples"] = code_dataset.run(self.llm, text)

        if dataset_type == "rag":
            context = brain_decision.get("context", "")
            result["rag_answer"] = rag_generator.run(self.llm, text, context)

        # =====================================================
        # 🔹 FINAL CONFIDENCE SCORE
        # =====================================================
        result["confidence_score"] = self.confidence.compute(result)

        return result