from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class SemanticBrainConfig(BaseSettings):
    # API Keys
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Model Selection
    primary_model: str = "llama-3.1-8b-instant"
    fallback_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_BRAIN_",
        env_file=".env",
        extra="ignore"
    )