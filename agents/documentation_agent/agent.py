from core.llm_provider import LLMProvider
from core.logger import logger
from .prompt import SYSTEM_PROMPT, build_prompt
from .config import DocumentationConfig


class DocumentationAgent:

    def __init__(self):
        self.config = DocumentationConfig()
        self.llm = LLMProvider(
            primary_model=self.config.primary_model,
            fallback_model=self.config.fallback_model,
            temperature=self.config.temperature
        )

    def run(self, dataset_info: dict) -> str:
        logger.info("Running Documentation Agent")

        user_prompt = build_prompt(dataset_info)

        readme = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        return readme