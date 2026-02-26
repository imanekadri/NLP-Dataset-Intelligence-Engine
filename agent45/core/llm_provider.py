import os
import anthropic
from openai import OpenAI
from agent45.core.logger import logger
from dotenv import load_dotenv

load_dotenv()


class LLMProvider:
    def __init__(self, primary_model="llama-3.1-8b-instant", fallback_model="claude-haiku-4-5-20251001", temperature=0):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.temperature = temperature

        # 1. Setup Primary Client (Llama via Groq)
        # Groq uses the OpenAI SDK, just with a different URL!
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            logger.warning("GROQ_API_KEY not found. Primary model will fail.")

        self.groq_client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        ) if groq_key else None

        # 2. Setup Fallback Client (Claude via Anthropic)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            logger.warning("ANTHROPIC_API_KEY not found. Fallback will not be available.")

        self.anthropic_client = anthropic.Anthropic(
            api_key=anthropic_key
        ) if anthropic_key else None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Attempt 1: Try Llama on Groq
        if self.groq_client:
            try:
                logger.info(f"Generating with primary model: {self.primary_model} (Groq)")
                response = self.groq_client.chat.completions.create(
                    model=self.primary_model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},  # Force JSON output
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq API failed with error: {e}. Initiating fallback...")

        # Attempt 2: Fallback to Claude
        if self.anthropic_client:
            try:
                logger.info(f"Generating with fallback model: {self.fallback_model} (Anthropic)")
                # Anthropic API format is slightly different: system prompt is a top-level parameter
                response = self.anthropic_client.messages.create(
                    model=self.fallback_model,
                    temperature=self.temperature,
                    max_tokens=2048,  # Anthropic requires max_tokens to be specified
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Fallback Anthropic API also failed: {e}")
                raise RuntimeError("Critical Failure: Both Llama and Claude models failed.")

        raise ValueError("No LLM clients were successfully configured. Check your API keys.")