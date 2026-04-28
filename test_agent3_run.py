import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent3_und import run_agent3

# Document de test minimal — texte inline (sans fichier)
state = {
    "raw_docs": [
        {
            "doc_id": "test_doc.txt",
            "topic": "machine_learning",
            "key_words": ["Python, TensorFlow, neural network, training, model"],
            "cleaned_text": (
                "TensorFlow is an excellent open-source framework developed by Google.\n\n"
                "It allows building deep learning models with Python.\n\n"
                "However, the documentation can be poor for beginners and some APIs are bad."
            ),
        }
    ]
}

result = run_agent3(state)

# Vérifier la sortie
import json
report_path = result["agent3_results"]
with open(report_path) as f:
    print(json.dumps(json.load(f), indent=2))
