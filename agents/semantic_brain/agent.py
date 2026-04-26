from core.llm_provider import LLMProvider
from .config import SemanticBrainConfig
from .prompt import SYSTEM_PROMPT, build_prompt
from .validator import validate_output
# from .domain_detector import DomainDetector
# from .ner_analyzer import NERAnalyzer
# from .topic_modeler import TopicModeler
from core.logger import logger


class DatasetBrainAgent:
    def __init__(self):
        # Load all settings once
        self.config = SemanticBrainConfig()

        # Pass the config to the provider
        self.llm = LLMProvider(
            primary_model=self.config.primary_model,
            fallback_model=self.config.fallback_model,
            temperature=self.config.temperature
        )

        # self.domain_detector = DomainDetector()
        # self.ner_analyzer = NERAnalyzer()
        # self.topic_modeler = TopicModeler()

    def run(self, analysis_summary: dict) -> dict:
        logger.info("Running Semantic Brain Agent")

        user_prompt = build_prompt(analysis_summary)
        raw_output = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        logger.debug(f"RAW AI OUTPUT: {raw_output}")

        validated = validate_output(raw_output)

        return validated.model_dump()

    # def run(self, metadata: dict, sample_texts: list) -> dict:

        # detected_domain = self.domain_detector.detect(sample_texts)
        # ner_stats = self.ner_analyzer.analyze(sample_texts)
        # topics = self.topic_modeler.extract_topics(sample_texts)
        #
        # enriched_metadata = {
        #     **metadata,
        #     "embedding_detected_domain": detected_domain,
        #     "ner_distribution": ner_stats,
        #     "dominant_topics": topics
        # }
        #
        # user_prompt = build_prompt(enriched_metadata)
        #
        # # The agent doesn't care if it's Llama or Claude, the provider handles it!
        # raw_output = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        # print(f"DEBUG - RAW AI OUTPUT: {raw_output}")
        # validated = validate_output(raw_output)
        #
        # return validated.model_dump()
        # # return user_prompt