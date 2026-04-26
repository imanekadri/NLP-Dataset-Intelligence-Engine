import os
import json
from collections import Counter

base_dir = r"c:\Users\KM-USER\Downloads\nlp\output\agent3\categories"
all_persons = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Les entités sont dans chaque paragraphe
                    paragraphs = data.get("paragraphs", [])
                    for p in paragraphs:
                        entities = p.get("entities", [])
                        for ent in entities:
                            if ent.get("label") == "PERSON":
                                all_persons.append(ent.get("text"))
            except:
                continue

counts = Counter(all_persons)
print("\n--- ANALYSE DES ERREURS PERSON (Agent 3) ---")
# Afficher les termes suspects (qui contiennent des mots qui ne sont pas des noms)
suspicious_words = ["Foreclosure", "Accountant", "Manager", "Analysis", "Data", "Report", "Project", "Lead", "Coordinate", "Assistant"]

for text, count in sorted(counts.items()):
    # On affiche tout pour que l'utilisateur puisse voir
    print(f"{{ \"text\": \"{text}\", \"label\": \"PERSON\" }},  ({count} fois)")
