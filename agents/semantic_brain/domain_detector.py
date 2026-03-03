from sentence_transformers import SentenceTransformer
import numpy as np

class DomainDetector:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.domain_templates = {
            "news": "politics world breaking news government",
            "resume": "experience skills education cv career",
            "email": "hello regards email message reply",
            "finance": "bank loan credit investment money",
            "medical": "patient diagnosis treatment hospital"
        }

        self.template_embeddings = {
            domain: self.model.encode(text)
            for domain, text in self.domain_templates.items()
        }

    def detect(self, sample_texts):
        embeddings = self.model.encode(sample_texts)
        avg_embedding = np.mean(embeddings, axis=0)

        best_domain = None
        best_score = -1

        for domain, template_emb in self.template_embeddings.items():
            score = np.dot(avg_embedding, template_emb)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain
