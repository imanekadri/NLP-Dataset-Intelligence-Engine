# -*- coding: utf-8 -*-
import sys
import os

print("🔍 Chemin actuel:", os.getcwd())
print("📁 Vérification du dossier data:")
data_path = os.path.join("agent1", "datasets", "data")
if os.path.exists(data_path):
    fichiers = os.listdir(data_path)
    print(f"✅ Dossier trouvé! Fichiers: {fichiers}")
else:
    print(f"❌ Dossier {data_path} non trouvé!")
    # Crée le dossier
    os.makedirs(data_path, exist_ok=True)
    print(f"✅ Dossier créé: {data_path}")

try:
    from agent1.agent import TextIngestionAgent
    print("✅ Import réussi!")
except Exception as e:
    print(f"❌ Erreur import: {e}")
    print("📋 Contenu du dossier courant:")
    print(os.listdir("."))
    sys.exit(1)

def main():
    print("🚀 Test Agent 1")
    
    # Initialise l'agent
    agent = TextIngestionAgent()
    
    # Scan le dossier data
    report = agent.scan_datasets()
    
    # Affiche les résultats
    print(f"\n📊 RÉSULTATS:")
    print(f"📁 Total documents: {report['stats']['total_docs']}")
    print(f"📋 Formats: {report['stats']['formats']}")
    print(f"🌍 Langues: {report['stats']['languages']}")

if __name__ == "__main__":
    main()