from pathlib import Path
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

    def run(self, dataset_info: dict) -> dict:
        logger.info("Running Documentation Agent")

        prompt = build_prompt(dataset_info)

        readme_content = self.llm.generate(
            SYSTEM_PROMPT,
            prompt
        )

        # Save README.md
        output_path = Path("output/agent9/README.md")
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        logger.info(f"README generated at {output_path}")

        return {
            "readme_path": str(output_path),
            "readme_content": readme_content
        }