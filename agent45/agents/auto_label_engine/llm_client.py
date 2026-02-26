import requests
from .config import GROQ_API_KEY, PRIMARY_MODEL, TEMPERATURE, GROQ_BASE_URL


class GroqClient:

    def generate(self, prompt: str, max_tokens=512):

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": PRIMARY_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens
        }

        response = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]