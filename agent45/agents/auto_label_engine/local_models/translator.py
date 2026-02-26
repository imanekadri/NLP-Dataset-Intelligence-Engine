from transformers import pipeline

class Translator:

    def __init__(self):
        self.model = pipeline("translation_fr_to_en",
                              model="Helsinki-NLP/opus-mt-fr-en")

    def translate(self, text):
        return self.model(text)[0]["translation_text"]