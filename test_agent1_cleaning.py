"""
Script de test pour Agent 1 : Ingestion + Cleaning
Lance seulement Agent 1 et affiche un avant/après du nettoyage.
"""
import os
import sys

# Ajouter le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔬 TEST AGENT 1 : INGESTION + CLEANING")
print("=" * 60)

# ---- ÉTAPE 1 : Ingestion ----
print("\n📥 ÉTAPE 1 : Ingestion des données...")
from agents.agent1_Ingestion.agent1_ingestion import run_ingestion

state = {}
state = run_ingestion(state)

print(f"\n✅ {len(state.get('raw_docs', []))} documents ingérés")

# ---- ÉTAPE 2 : Cleaning ----
print("\n🧹 ÉTAPE 2 : Nettoyage des données...")
from agents.agent1_Ingestion.agent1_cleaning import run_cleaning

state = run_cleaning(state)

# ---- ÉTAPE 3 : Vérification avant/après ----
print("\n" + "=" * 60)
print("📊 RÉSULTATS : AVANT vs APRÈS nettoyage")
print("=" * 60)

cleaned_count = 0
for doc in state.get("raw_docs", []):
    if not doc.get("is_cleaned"):
        continue
    cleaned_count += 1
    
    doc_id = doc.get("doc_id", "?")
    extracted_path = doc.get("extracted_path", "")
    cleaned_path = doc.get("cleaned_text_path", "")
    
    # Lire les deux versions
    raw_text = ""
    cleaned_text = ""
    
    if extracted_path and os.path.exists(extracted_path):
        with open(extracted_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    
    if cleaned_path and os.path.exists(cleaned_path):
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()
    
    # Afficher un résumé pour les 5 premiers documents
    if cleaned_count <= 5:
        print(f"\n{'─' * 50}")
        print(f"📄 {doc_id} | Format: {doc.get('format', '?')} | Langue: {doc.get('lang', '?')}")
        print(f"   Taille AVANT : {len(raw_text):,} chars | {len(raw_text.split()):,} mots")
        print(f"   Taille APRÈS : {len(cleaned_text):,} chars | {len(cleaned_text.split()):,} mots")
        reduction = (1 - len(cleaned_text) / max(len(raw_text), 1)) * 100
        print(f"   🗑️  Bruit supprimé : {reduction:.1f}%")
        
        # Montrer un extrait du texte nettoyé
        preview = cleaned_text[:200].replace('\n', ' ')
        print(f"   📝 Aperçu nettoyé : \"{preview}...\"")

print(f"\n{'=' * 60}")
print(f"✅ TOTAL : {cleaned_count} documents nettoyés avec succès")
print(f"📁 Fichiers nettoyés dans : output/agent1/cleaned/")
print(f"{'=' * 60}")
