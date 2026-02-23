import re
from pipeline.state import NLPPipelineState
from pipeline.config import PipelineConfig


def _detect_pii(text):
    patterns = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone": r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    }
    for pattern in patterns.values():
        if re.search(pattern, text):
            return True
    return False


def run_understanding(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 3 — NLP Understanding Engine.
    Embeddings (déjà dans state via Agent 2), NER (spaCy), Sentiment (transformers), PII.
    """
    config = PipelineConfig()
    texts = state.get("texts", [])
    raw_docs = state.get("raw_docs", [])

    if not texts:
        print("[Agent 3] No texts in state, skipping.")
        return state

    print(f"[Agent 3] Processing {len(texts)} texts...")

    # --- NER via spaCy ---
    print("[Agent 3] Loading spaCy model...")
    import spacy
    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        print(f"[Agent 3] spaCy model '{config.SPACY_MODEL}' not found. Run: python -m spacy download {config.SPACY_MODEL}")
        nlp = None

    ner_results = {}
    for i, text in enumerate(texts):
        doc_id = raw_docs[i]["doc_id"] if i < len(raw_docs) else f"text_{i}"
        if nlp:
            doc = nlp(text[:config.MAX_TEXT_LENGTH_NLP * 5])
            entities = [
                {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents
            ]
            ner_results[doc_id] = entities
        else:
            ner_results[doc_id] = []

    # --- Sentiment Analysis via transformers ---
    print("[Agent 3] Loading sentiment model...")
    from transformers import pipeline as hf_pipeline
    try:
        sentiment_pipe = hf_pipeline(
            "sentiment-analysis",
            model=config.SENTIMENT_MODEL,
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH_NLP,
        )
    except Exception as e:
        print(f"[Agent 3] Sentiment model failed to load: {e}")
        sentiment_pipe = None

    sentiment_results = {}
    for i, text in enumerate(texts):
        doc_id = raw_docs[i]["doc_id"] if i < len(raw_docs) else f"text_{i}"
        if sentiment_pipe and text.strip():
            try:
                result = sentiment_pipe(text[:config.MAX_TEXT_LENGTH_NLP])[0]
                sentiment_results[doc_id] = result["label"].lower()
            except Exception:
                sentiment_results[doc_id] = "unknown"
        else:
            sentiment_results[doc_id] = "unknown"

    # --- PII Detection ---
    print("[Agent 3] Detecting PII...")
    pii_flags = {}
    for i, text in enumerate(texts):
        doc_id = raw_docs[i]["doc_id"] if i < len(raw_docs) else f"text_{i}"
        has_pii = _detect_pii(text)
        # Vérifier aussi les entités PERSON dans NER
        if not has_pii and ner_results.get(doc_id):
            has_pii = any(e["label"] == "PERSON" for e in ner_results[doc_id])
        pii_flags[doc_id] = has_pii

    pii_count = sum(1 for v in pii_flags.values() if v)
    print(f"[Agent 3] NER: {sum(len(v) for v in ner_results.values())} entities, "
          f"PII: {pii_count} docs flagged.")

    return {
        **state,
        "ner_results": ner_results,
        "sentiment_results": sentiment_results,
        "pii_flags": pii_flags,
    }
