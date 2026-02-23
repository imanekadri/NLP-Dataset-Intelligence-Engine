# NLP Dataset Intelligence Engine

Pipeline multi-agent **LangGraph** pour l'ingestion, le profilage et l'export intelligent de datasets NLP.

---

## Architecture

9 agents autonomes orchestrés par LangGraph, partageant un **SharedState** unique :

```
START
  │
  ▼
┌─────────────────┐
│  Agent 1        │  Ingestion : scan + extract + trace CSV + lang detect
│  Ingestion      │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 2        │  Profiler : BERTopic + clustering + structure + anomalies
│  Profiler NLP   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 3        │  NLP Understanding : embeddings + NER + sentiment + PII
│  Understanding  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 4        │  Semantic Brain : type dataset + format ML (heuristique/LLM)
│  Semantic Brain │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 5        │  Auto Label : intent + labels (lit NER/sentiment du state)
│  Auto Label     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 6        │  Dataset Architect : structure train/val/test + JSONL
│  Architect      │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 7        │  Export : HuggingFace, JSONL, CSV, OpenAI fine-tune
│  Export Engine  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 8        │  QA : toxicité, biais, équilibre classes, PII check
│  QA Agent       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Agent 9        │  README Generator : documentation auto complète
│  README Gen     │
└────────┬────────┘
         ▼
        END
```

**Principe** : chaque agent lit et enrichit le SharedState — aucune donnée n'est recalculée.

---

## Zéro duplication

| Opération | Calculée par | Réutilisée par |
|---|---|---|
| Détection de langue | Agent 1 | Agents 2, 4, 8 |
| Embeddings | Agent 2 | Agents 3, 5, 6 |
| BERTopic | Agent 2 | Agents 4, 5 |
| NER (spaCy) | Agent 3 | Agent 5 |
| Sentiment | Agent 3 | Agents 5, 8 |
| PII | Agent 3 | Agent 8 |

---

## Installation

### Prérequis système

- **Python 3.9+**
- **Tesseract OCR** (pour les PDFs scannés) : [Télécharger](https://github.com/UB-Mannheim/tesseract/wiki)
  - Chemin attendu : `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Poppler** (pour la conversion PDF vers image) : [Télécharger](https://github.com/oschwartz10612/poppler-windows/releases/)
  - Chemin attendu : `C:\Program Files\poppler-25.12.0\Library\bin`

### Créer et activer le virtual environment

```bash
# Création
py -m venv venv

# Activation (Windows - PowerShell)
venv\Scripts\Activate

# Activation (Git Bash)
source venv/Scripts/activate
```

### Installer les dépendances

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Utilisation

### Lancer le pipeline complet

```bash
# Depuis la racine du projet
python -m pipeline.run
```

Le pipeline exécute les 9 agents séquentiellement et produit tous les outputs dans `output/`.

### Structure des sorties

```
output/
├── extracted/          # Textes extraits (Agent 1)
├── traces/             # trace_index.csv, documents_info.json, report.json
├── profiles/           # dataset_profile.json, clusters.json, topics.json
├── exports/            # dataset.jsonl, dataset.csv, huggingface/, openai_finetune.jsonl
└── reports/            # qa_report.json, README_dataset.md, final_state_summary.json
```

---

## Structure du projet

```
NLP-Dataset-Intelligence-Engine/
├── pipeline/
│   ├── state.py               # NLPPipelineState (SharedState TypedDict)
│   ├── config.py              # Configuration unifiée
│   ├── graph.py               # Définition du graphe LangGraph
│   └── run.py                 # Point d'entrée
├── agents/
│   ├── agent1_ingestion.py    # Scan + Extract + Trace CSV + Lang detect
│   ├── agent2_profiler.py     # BERTopic + Clustering + Stats + Noise
│   ├── agent3_understanding.py # NER + Sentiment + PII
│   ├── agent4_semantic_brain.py # Inférence type dataset + format ML
│   ├── agent5_auto_label.py   # Intent + Labels automatiques
│   ├── agent6_dataset_architect.py # Splits train/val/test
│   ├── agent7_export_engine.py # Export multi-format
│   ├── agent8_qa_agent.py     # Qualité, biais, PII
│   └── agent9_readme_generator.py # README auto-généré
├── utils/
│   ├── extraction.py          # Extraction de texte (PDF, DOCX, CSV...)
│   └── text_analysis.py       # Analyse de bruit, structure, langue
├── models/
│   └── all-MiniLM-L6-v2/     # Modèle d'embedding local
├── data/                      # Données d'entrée
├── output/                    # Résultats du pipeline
├── requirements.txt
└── README.md
```

---

## SharedState

Tous les agents lisent et écrivent dans un seul objet d'état :

```python
class NLPPipelineState(TypedDict, total=False):
    # Agent 1
    raw_docs, trace_csv_path, extracted_files, languages_detected, duplicates_removed
    # Agent 2
    texts, embeddings, topic_clusters, dataset_profile
    # Agent 3
    ner_results, sentiment_results, pii_flags
    # Agent 4
    dataset_type, ml_format, tasks, labeling_strategy
    # Agent 5
    labels
    # Agent 6
    splits, formatted_dataset
    # Agent 7
    export_paths
    # Agent 8
    quality_score, qa_report
    # Agent 9
    readme_content
```

---

## Description des Agents

### Agent 1 — Data Ingestion
- Scan récursif de `data/`
- Extraction multi-format : PDF (+ OCR), CSV, JSON, XML, DOCX, TXT
- Déduplication MD5
- Détection de langue (langdetect) — **une seule fois**
- Traçabilité : `trace_index.csv` + `documents_info.json`

### Agent 2 — Profiler NLP
- Analyse statistique (longueur, vocabulaire)
- Détection du bruit (ratio de caractères spéciaux)
- Détection de structure (Code / Dialogue / Paragraphe)
- Embeddings sémantiques (`all-MiniLM-L6-v2`) — **une seule fois**
- Clustering KMeans
- Topic Modeling (BERTopic)

### Agent 3 — NLP Understanding
- NER via spaCy (`en_core_web_sm`) — **une seule fois**
- Sentiment via transformers (`distilbert-base-uncased`) — **une seule fois**
- PII Detection (regex + entités PERSON)

### Agent 4 — Semantic Brain
- Inférence heuristique du type de dataset (NER, sentiment, conversationnel, code...)
- Détermine le format ML optimal (classification, token_classification, instruction_tuning)
- Définit la stratégie de labeling

### Agent 5 — Auto Label
- Lit NER, sentiment, topics depuis le state (**pas de recalcul**)
- Génère des labels composites : intent, catégorie, sentiment, entités

### Agent 6 — Dataset Architect
- Split train/val/test (80/10/10)
- Format JSONL structuré avec métadonnées

### Agent 7 — Export Engine
- JSONL complet
- CSV aplati
- HuggingFace format (fichiers séparés par split)
- OpenAI fine-tune format

### Agent 8 — QA Agent
- Équilibre des classes
- Vérification PII
- Biais linguistique
- Complétude des labels
- Score de qualité (0-100)

### Agent 9 — README Generator
- README Markdown auto-généré depuis le state final
- Inclut overview, labels, splits, exports, qualité

---

## Configuration

Toute la configuration est centralisée dans `pipeline/config.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `DATASETS_DIR` | `data/` | Dossier source des données |
| `NUM_CLUSTERS` | `5` | Nombre de clusters KMeans |
| `NOISE_THRESHOLD` | `0.30` | Seuil de bruit |
| `USE_TOPIC_MODELING` | `True` | Activer BERTopic |
| `SPACY_MODEL` | `en_core_web_sm` | Modèle spaCy pour NER |
| `TRAIN_RATIO` | `0.8` | Ratio de split train |

---

## Stack technique

| Composant | Technologie |
|---|---|
| Orchestration | LangGraph |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| NER | spaCy |
| Sentiment | transformers (DistilBERT) |
| Topics | BERTopic |
| Clustering | scikit-learn (KMeans) |
| Extraction PDF | PyPDF2 + Tesseract OCR |
| Language detect | langdetect |
