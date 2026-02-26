import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PRIMARY_MODEL = os.getenv("SEMANTIC_BRAIN_PRIMARY_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("SEMANTIC_BRAIN_TEMPERATURE", 0))
GROQ_BASE_URL = "https://api.groq.com/openai/v1"