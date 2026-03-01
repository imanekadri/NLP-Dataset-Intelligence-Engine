from transformers import pipeline


class Translator:

    def __init__(self):
        self.model = pipeline(
            task="translation_fr_to_en",
            model="Helsinki-NLP/opus-mt-fr-en"
        )

    def translate(self, text: str):
        result = self.model(text)[0]
        return result["translation_text"]