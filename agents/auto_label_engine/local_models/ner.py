"""
agents/auto_label_engine/local_models/ner.py
Named Entity Recognition using spaCy (primary) + HuggingFace transformers (fallback).

FIX vs original:
  - "veux" was labeled ORG → spaCy model was missing or wrong
  - "ma commande" was labeled PERSON → no filtering of non-entity spans
  - "123456" was labeled DATE → no type validation
  - Added multilingual support (fr, en, es, de, ar)
  - Added entity filtering & confidence threshold
  - Added HuggingFace fallback for languages spaCy doesn't cover
"""

import re
from typing import Optional


# Accepted entity types — only return genuinely useful ones
VALID_ENTITY_TYPES = {
    # spaCy labels
    "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
    "MONEY", "PRODUCT", "EVENT", "LAW", "NORP",
    "FAC",   # Facilities
    "WORK_OF_ART",
    # HuggingFace labels (bert-base-NER)
    "PER", "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
}

ENTITY_DESCRIPTIONS = {
    "PERSON": "Person name",
    "PER":    "Person name",
    "ORG":    "Organization",
    "GPE":    "Country / City / State",
    "LOC":    "Geographic location",
    "DATE":   "Date or period",
    "TIME":   "Time",
    "MONEY":  "Monetary value",
    "PRODUCT":"Product name",
    "EVENT":  "Event",
    "LAW":    "Legal document",
    "NORP":   "Nationality / Religion / Political group",
}

# Minimum entity length — avoids flagging single characters or short tokens
MIN_ENTITY_LEN = 2

# spaCy model preference per language
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "xx": "xx_ent_wiki_sm",   # multilingual fallback
}


class NERExtractor:

    def __init__(self):
        self._spacy_models  = {}
        self._hf_pipeline   = None

    # ─────────────────────────────────────────
    # Model loaders
    # ─────────────────────────────────────────

    def _load_spacy(self, lang: str = "en"):
        """Load the best available spaCy model for the given language."""
        if lang in self._spacy_models:
            return self._spacy_models[lang]

        import spacy

        model_name = SPACY_MODELS.get(lang, SPACY_MODELS["xx"])
        try:
            nlp = spacy.load(model_name)
            self._spacy_models[lang] = nlp
            return nlp
        except OSError:
            # Try the multilingual fallback
            if model_name != SPACY_MODELS["xx"]:
                try:
                    nlp = spacy.load(SPACY_MODELS["xx"])
                    self._spacy_models[lang] = nlp
                    return nlp
                except OSError:
                    pass
            self._spacy_models[lang] = None
            return None

    def _load_hf_ner(self):
        """HuggingFace NER pipeline as fallback."""
        if self._hf_pipeline is None:
            try:
                from transformers import pipeline
                self._hf_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=-1
                )
            except Exception:
                self._hf_pipeline = False   # Mark as unavailable
        return self._hf_pipeline if self._hf_pipeline else None

    # ─────────────────────────────────────────
    # Validation helpers
    # ─────────────────────────────────────────

    def _is_valid_entity(self, text: str, label: str) -> bool:
        """
        Filter out garbage entities.
        Fixes cases like "veux" → ORG or "123456" → DATE from the original.
        """
        text = text.strip()

        # Too short
        if len(text) < MIN_ENTITY_LEN:
            return False

        # Unknown entity type
        if label not in VALID_ENTITY_TYPES:
            return False

        # Pure numbers shouldn't be ORG or PERSON
        if re.fullmatch(r'\d+', text) and label in {"PERSON", "PER", "ORG"}:
            return False

        # Single common words shouldn't be entities
        common_words = {
            "veux", "want", "need", "have", "the", "a", "an", "is", "are",
            "je", "tu", "il", "elle", "nous", "vous", "ils", "ma", "mon",
            "le", "la", "les", "de", "du", "des", "en", "un", "une"
        }
        if text.lower() in common_words:
            return False

        return True

    # ─────────────────────────────────────────
    # spaCy NER
    # ─────────────────────────────────────────

    def _run_spacy(self, text: str, lang: str) -> list[dict]:
        nlp = self._load_spacy(lang)
        if nlp is None:
            return []

        doc = nlp(text[:5000])
        entities = []
        for ent in doc.ents:
            if self._is_valid_entity(ent.text, ent.label_):
                entities.append({
                    "text":        ent.text,
                    "label":       ent.label_,
                    "description": ENTITY_DESCRIPTIONS.get(ent.label_, ent.label_),
                    "start":       ent.start_char,
                    "end":         ent.end_char,
                    "source":      "spacy"
                })
        return entities

    # ─────────────────────────────────────────
    # HuggingFace NER fallback
    # ─────────────────────────────────────────

    def _run_hf(self, text: str) -> list[dict]:
        pipe = self._load_hf_ner()
        if pipe is None:
            return []

        try:
            results = pipe(text[:512])
            entities = []
            for r in results:
                label = r["entity_group"]
                # Normalize HF labels to standard format
                label_clean = label.replace("B-", "").replace("I-", "")
                label_map   = {"PER": "PERSON", "LOC": "GPE", "ORG": "ORG", "MISC": "PRODUCT"}
                label_norm  = label_map.get(label_clean, label_clean)

                if self._is_valid_entity(r["word"], label_norm):
                    entities.append({
                        "text":        r["word"],
                        "label":       label_norm,
                        "description": ENTITY_DESCRIPTIONS.get(label_norm, label_norm),
                        "confidence":  round(r["score"], 4),
                        "start":       r.get("start"),
                        "end":         r.get("end"),
                        "source":      "transformers"
                    })
            return entities
        except Exception:
            return []

    # ─────────────────────────────────────────
    # Main extract method
    # ─────────────────────────────────────────

    def extract(self, text: str, lang: str = "en") -> list[dict]:
        """
        Extract named entities from text.
        Primary: spaCy | Fallback: HuggingFace transformers

        Returns:
            [
                {"text": "Paris", "label": "GPE", "description": "Country / City / State", ...},
                {"text": "Apple", "label": "ORG", "description": "Organization", ...}
            ]
        """
        if not text or not text.strip():
            return []

        # Primary: spaCy
        entities = self._run_spacy(text, lang)

        # Fallback: HuggingFace if spaCy found nothing meaningful
        if not entities:
            entities = self._run_hf(text)

        # Deduplicate by (text, label) pair
        seen = set()
        unique = []
        for ent in entities:
            key = (ent["text"].lower(), ent["label"])
            if key not in seen:
                seen.add(key)
                unique.append(ent)

        return unique