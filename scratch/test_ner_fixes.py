# import spacy
from core.utils import extract_entities
from core.config import Agent3Config

def test_ner_logic():
    # Mock models (simulated)
    # Since I can't easily load real spacy models in a 5s script, 
    # I will test the logic by mocking the doc.ents
    
    class MockEnt:
        def __init__(self, text, label):
            self.text = text
            self.label_ = label

    class MockDoc:
        def __init__(self, ents):
            self.ents = ents

    class MockNLP:
        def __call__(self, text):
            # Simulate what a noisy model might return
            if "Alan Turing" in text:
                return MockDoc([
                    MockEnt("Alan Turing", "PERSON"),
                    MockEnt("Python", "ORG"),
                    MockEnt("Example Training", "PERSON"),
                    MockEnt("Big Data Big Data", "ORG")
                ])
            return MockDoc([])

    config = Agent3Config()
    nlp = MockNLP()
    
    text = "Alan Turing used Python for Example Training on Big Data Big Data."
    
    # Run our refined extraction
    results = extract_entities(
        text, 
        "en", 
        nlp, nlp, 
        valid_labels=config.VALID_ENTITY_LABELS,
        tech_keywords=config.TECH_KEYWORDS,
        generic_blacklist=config.GENERIC_WORDS_BLACKLIST
    )
    
    print("Test Results:")
    for t, l in results:
        print(f"- {t}: {l}")

    # Expected: 
    # Alan Turing: PERSON
    # Python: TECH
    # Example Training: DISCARDED (Generic/Verb-like)
    # Big Data: ORG (Cleaned and Deduplicated)

if __name__ == "__main__":
    test_ner_logic()
