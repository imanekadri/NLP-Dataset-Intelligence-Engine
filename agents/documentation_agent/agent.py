from core.llm_provider import LLMProvider
from core.logger import logger
from .prompt import SYSTEM_PROMPT, build_prompt


class DocumentationAgent:

    def __init__(self):
        self.llm = LLMProvider(
            primary_model="llama-3.1-8b-instant",
            temperature=0.2
        )

    def run(self, dataset_info: dict) -> str:
        logger.info("Running Documentation Agent")

        user_prompt = build_prompt(dataset_info)

        readme = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        return readme