"""
Script de test pour Agent 3 : Sentiment + Entités (TECH vs PERSON)
Vérifie que le nettoyage et la détection fonctionnent ensemble.
"""
import os
import sys
import spacy
from transformers import pipeline

# Ajouter le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent1_Ingestion.agent1_cleaning import clean_text_pipeline
from core.utils import analyze_sentiment, extract_entities
from core.config import agent3_config as config

print("=" * 60)
print("🎭 TEST AGENT 3 : SENTIMENTS & ENTITÉS")
print("=" * 60)

# 1. Charger les modèles (comme le fait l'Agent 3)
print("\n🤖 Chargement des modèles...")
nlp_en = spacy.load(config.SPACY_EN)
sentiment_model = pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL, truncation=True)

# 2. Phrases de test
test_phrases = [
    "In [1]: Jupyter Notebook is an excellent tool for data science!",
    "I have a big problem with my Python installation on Windows C:\\Users\\Admin.",
    "Guido van Rossum created Python, it is a very useful language.",
    "The analysis of the data was poor and the results are bad."
]

print("\n" + "=" * 60)
print(f"{'PHRASE ORIGINALE':<40} | {'SENTIMENT':<10} | {'ENTITÉS'}")
print("-" * 60)

for phrase in test_phrases:
    # A. Nettoyage (Agent 1)
    cleaned = clean_text_pipeline(phrase)
    
    # B. Sentiment (Agent 3)
    # On utilise les triggers de la config
    sentiment = analyze_sentiment(sentiment_model, cleaned, config.SENTIMENT_TRIGGER_KEYWORDS)
    
    # C. Entités (Agent 3)
    entities = extract_entities(
        cleaned, "en", nlp_en, nlp_en, 
        valid_labels=config.VALID_ENTITY_LABELS,
        tech_keywords=config.TECH_KEYWORDS,
        generic_blacklist=config.GENERIC_WORDS_BLACKLIST
    )
    
    # Formatage des entités pour l'affichage
    ent_list = [f"{e['text']} ({e['label']})" for e in entities]
    ent_str = ", ".join(ent_list) if ent_list else "Aucune"
    
    # Affichage
    short_raw = (phrase[:37] + '..') if len(phrase) > 37 else phrase
    print(f"{short_raw:<40} | {sentiment:<10} | {ent_str}")

print("=" * 60)
print("✅ Test terminé. Notez que 'Jupyter Notebook' doit être 'TECH' et non 'PERSON'.")
