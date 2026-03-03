import spacy
from collections import Counter

class NERAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, texts):
        entity_counter = Counter()

        for text in texts:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_counter[ent.label_] += 1

        total_entities = sum(entity_counter.values())

        if total_entities == 0:
            return {}

        distribution = {
            label: count / total_entities
            for label, count in entity_counter.items()
        }

        return distribution
