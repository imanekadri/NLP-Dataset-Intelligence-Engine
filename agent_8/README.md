# 🔍 Agent 8 – Quality Gate

> **Rôle dans le pipeline :** Agent 8 est le contrôleur de qualité final du projet *NLP Dataset Intelligence Engine*. Il analyse les données produites par les agents précédents et génère un **score de qualité global (0–100)** accompagné de **recommandations d'amélioration** concrètes.

---

## 📌 Table des matières

1. [Présentation](#-présentation)
2. [Structure du dossier](#-structure-du-dossier)
3. [Inputs (données reçues)](#-inputs-données-reçues)
4. [Analyses effectuées](#️-analyses-effectuées)
5. [Calcul du Quality Score](#-calcul-du-quality-score)
6. [Outputs (données produites)](#-outputs-données-produites)
7. [Comment exécuter l'agent](#-comment-exécuter-lagent)
8. [Intégration avec les autres agents](#-intégration-avec-les-autres-agents)
9. [Dépendances](#-dépendances)

---

## 📋 Présentation

Agent 8 joue le rôle de **porte de qualité** (*quality gate*) dans le pipeline multi-agents. Il reçoit les résultats des analyses précédentes et répond à la question :

> *"Le dataset est-il suffisamment propre pour entraîner un modèle de machine learning ?"*

Il effectue 4 types d'analyses automatiques et produit :
- Un **score global de qualité** entre 0 et 100.
- Un **rapport JSON** structuré (exploitable par d'autres programmes).
- Un **rapport PDF** lisible par un humain, avec tableaux et graphiques.

---

## 📁 Structure du dossier

```
agent_8/
├── __init__.py            ← Expose run_agent8() pour l'intégration pipeline
├── main.py                ← Point d'entrée ; orchestre les 4 analyses
├── utils.py               ← Fonctions utilitaires (chargement JSON, logging, chemins)
├── quality_checks.py      ← Les 5 classes d'analyse de qualité
├── report_generator.py    ← Génération des rapports JSON et PDF
└── output/
    ├── quality_report.json   ← Rapport structuré (généré à l'exécution)
    └── quality_report.pdf    ← Rapport lisible (généré à l'exécution)
```

### Rôle de chaque fichier

| Fichier | Responsabilité |
|---|---|
| `main.py` | Orchestrateur : appelle les analyses dans l'ordre, collecte les résultats, génère les rapports |
| `quality_checks.py` | Contient les 5 classes d'analyse (toxicité, classes, PII, bruit, score) |
| `report_generator.py` | Génère `quality_report.json` et `quality_report.pdf` |
| `utils.py` | Fonctions partagées : `load_json`, `save_json`, `log_section`, chemins des fichiers |
| `__init__.py` | Expose `run_agent8()` pour que l'orchestrateur puisse importer l'agent |

---

## 📥 Inputs (données reçues)

Agent 8 reçoit ses données depuis les agents précédents du pipeline.

### De l'Agent 5 (AutoLabelEngine)
L'Agent 5 labellise chaque texte du dataset. Agent 8 utilise ses sorties pour analyser la toxicité et l'équilibre des classes.

| Clé dans `state` | Type | Description |
|---|---|---|
| `labeled_records` | `list[dict]` | Liste des enregistrements avec `text`, `intent`, `sentiment`, `topic_id` |
| `intents` | `list[str]` | Liste des labels d'intention (ex: `["cancel_order", "refund", ...]`) |
| `entities` | `list` | Entités nommées extraites (ex: `[{"text": "Paris", "type": "LOC"}]`) |
| `sentiments` | `list[str]` | Labels de sentiment (ex: `["positive", "negative", "neutral"]`) |

### De l'Agent 3 (PII Detector)
L'Agent 3 détecte les données personnelles. Agent 8 consomme son rapport pour calculer le taux de PII.

| Clé dans `state` | Type | Description |
|---|---|---|
| `pii_flags` | `list` ou `dict` | Rapport PII : liste d'entrées avec entités détectées, ou résumé global |

> ℹ️ **Format accepté :**
> - **Liste** : `[{"text": "...", "pii": [{"type": "EMAIL"}]}, ...]`
> - **Dict global** : `{"pii_count": 12, "types": ["EMAIL", "PHONE"]}`

### De l'Agent 2 (Profiler)
L'Agent 2 calcule un profil statistique du dataset. Agent 8 utilise `avg_noise_ratio` pour l'analyse du bruit.

| Clé dans `state` | Type | Description |
|---|---|---|
| `dataset_profile` | `dict` | Profil avec `avg_noise_ratio`, `dialogue_ratio`, `topics_detected`, etc. |

### Paramètre utilisateur
| Clé dans `state` | Type | Description |
|---|---|---|
| `topic` | `str` | Sujet principal attendu du dataset (ex: `"customer_support"`) |

---

## ⚙️ Analyses effectuées

### Analyse 1 – Toxicité (`ToxicityChecker`)

**Objectif :** Mesurer le pourcentage de textes négatifs ou toxiques, et identifier les textes hors-sujet.

**Méthode :**
1. **Via l'Agent 5** : Si un label de sentiment `"negative"` est disponible → le texte est compté comme toxique.
2. **Fallback (mots-clés)** : Si pas de sentiment → on cherche des mots toxiques dans le texte brut (*hate, violence, insulte, haine...*).
3. **Hors-sujet** : Un texte est hors-sujet si son `topic_id` contient `"unknown"` ou ne correspond pas au topic attendu.

**Seuil d'alerte :** `> 15 %` de textes toxiques, ou `> 20 %` de textes hors-sujet.

---

### Analyse 2 – Équilibre des classes (`ClassBalanceChecker`)

**Objectif :** Vérifier si les intents et sentiments sont équitablement répartis.

**Méthode :**
- On compte les occurrences de chaque classe (ex: `{"cancel": 300, "refund": 50}`).
- On calcule le **ratio de déséquilibre** = `classe_max / classe_min`.
- Si ce ratio dépasse **3.0** → dataset déséquilibré.

**Exemple :**
```
intents = {"cancel_order": 300, "refund": 50, "complaint": 150}
ratio = 300 / 50 = 6.0  →  DÉSÉQUILIBRÉ (6.0 > 3.0)
```

**Impact :** Un dataset très déséquilibré peut biaiser un modèle de ML (il sera bon pour la classe majoritaire, mauvais pour les autres).

---

### Analyse 3 – Données personnelles (`PIIChecker`)

**Objectif :** Détecter les données sensibles (emails, numéros de téléphone, numéros de carte bancaire) dans le dataset.

**Sources de détection (par ordre de priorité) :**
1. **Agent 3** (PII Detector basé sur Microsoft Presidio) → résultats les plus fiables.
2. **Regex locale** (fallback) → si Agent 3 n'est pas disponible, on applique des expressions régulières :
   - Email : `jean.dupont@gmail.com`
   - Téléphone : `+33 6 12 34 56 78`
   - Carte bancaire : `4539 1488 0343 6467`

**Seuil d'alerte :** `> 5 %` des entrées contiennent des PII.

---

### Analyse 4 – Bruit & Doublons (`NoiseAndDupChecker`)

**Objectif :** Détecter les données inutiles qui polluent le dataset.

**Détection des doublons :**
- On normalise chaque texte (minuscules + suppression des espaces multiples).
- On parcourt tous les textes en gardant une mémoire des textes vus.
- Tout texte déjà rencontré → doublon.

**Détection du bruit :**
1. **Agent 2** (priorité) → `avg_noise_ratio` déjà calculé par le Profiler.
2. **Analyse locale** (fallback) : un texte est du bruit si :
   - Il est vide ou trop court (< 5 mots).
   - Il ne contient aucune lettre alphabétique (`"12345 !!!??"`).

**Seuils d'alerte :** doublons `> 5 %`, bruit `> 10 %`.

---

## 📊 Calcul du Quality Score

Le score de qualité part de **100 points** et se réduit à chaque problème détecté.

```
Score = 100 - Σ(pénalités)
```

### Table des pénalités

| Problème détecté | Pénalité | Calcul |
|---|---|---|
| Toxicité élevée | max **−20 pts** | `toxicity_rate × 100` |
| Contenu hors-sujet | max **−10 pts** | `off_topic_rate × 50` |
| Déséquilibre de classes | **−15 pts** fixe | si `is_imbalanced = True` |
| Données PII sensibles | max **−15 pts** | `pii_rate × 100` |
| Doublons élevés | max **−15 pts** | `duplication_rate × 100` |
| Bruit élevé | max **−15 pts** | `noise_rate × 80` |

### Interprétation du score

| Score | Interprétation | Couleur dans le PDF |
|---|---|---|
| 80 – 100 | Bonne qualité ✅ | 🟢 Vert |
| 60 – 79 | Qualité acceptable, améliorable ⚠️ | 🟠 Orange |
| 0 – 59 | Mauvaise qualité, action urgente ❌ | 🔴 Rouge |

### Exemple de calcul

```
toxicity_rate   = 0.20  →  pénalité = min(0.20 × 100, 20) = 20 pts
is_imbalanced   = True  →  pénalité = 15 pts
duplication_rate = 0.08 →  pénalité = min(0.08 × 100, 15) = 8 pts
pii_rate        = 0.0   →  pénalité = 0 pts
noise_rate      = 0.0   →  pénalité = 0 pts
off_topic_rate  = 0.0   →  pénalité = 0 pts

Quality Score = 100 - 20 - 15 - 8 = 57 / 100  →  Mauvaise qualité
```

---

## 📤 Outputs (données produites)

### 1. `quality_report.json`

Rapport JSON structuré, exploitable par d'autres programmes ou agents.

```json
{
  "agent": "Agent8_QualityGate",
  "timestamp": "2026-04-19T11:36:53.603626",
  "topic": "customer_support",
  "quality_score": 85,
  "toxicity_rate": 0.12,
  "off_topic_rate": 0.05,
  "class_balance": "imbalanced",
  "intent_distribution": {"cancel_order": 150, "refund": 50, "complaint": 80},
  "sentiment_distribution": {"positive": 180, "negative": 100},
  "pii_detected": true,
  "pii_rate": 0.03,
  "pii_types": ["EMAIL", "PHONE"],
  "duplication_rate": 0.08,
  "noise_rate": 0.06,
  "issues": ["toxic content", "class imbalance", "high duplication rate"],
  "recommendations": [
    "Nettoyer ou supprimer les 34 textes toxiques/négatifs (12.0% du dataset).",
    "Rééquilibrer les classes (ratio déséquilibre : 3.0×). Utiliser SMOTE.",
    "Dédupliquer le dataset : 23 doublons détectés (8.0%)."
  ]
}
```

### 2. `quality_report.pdf`

Rapport PDF lisible, généré avec **reportlab** et **matplotlib**.

**Contenu :**
- En-tête avec date d'exécution et topic analysé
- Tableau coloré des métriques (score en vert/orange/rouge selon la valeur)
- Graphique en barres des 5 métriques principales
- Tableaux de distribution des intents et sentiments
- Liste des problèmes détectés
- Recommandations numérotées et personnalisées
- Pied de page

---

## ▶️ Comment exécuter l'agent

### Mode autonome (test standalone)

```bash
# Depuis la racine du projet
python -X utf8 agent_8/main.py
```

> Agent 8 va chercher automatiquement les fichiers JSON des agents précédents dans :
> - `output/agent2/dataset_profile.json`
> - `output/agent3/pii_report.json`
> - `output/agent5/labels.json`

### Mode pipeline (depuis un orchestrateur)

```python
from agent_8 import run_agent8

# Préparer l'état du pipeline
state = {
    "topic": "customer_support",
    "labeled_records": [
        {"text": "Je veux annuler ma commande", "intent": "cancel_order",
         "sentiment": {"label": "negative"}, "topic_id": "support"},
        {"text": "Bonjour, comment puis-je vous aider ?", "intent": "greeting",
         "sentiment": {"label": "positive"}, "topic_id": "support"},
        # ... autres enregistrements ...
    ],
    "intents": ["cancel_order", "greeting", ...],
    "sentiments": ["negative", "positive", ...],
    "pii_flags": [{"pii": []}, {"pii": [{"type": "EMAIL"}]}],
    "dataset_profile": {"avg_noise_ratio": 0.05, "dialogue_ratio": 0.6},
}

# Lancer Agent 8
result = run_agent8(state)

# Accéder aux résultats
print(result["quality_score"])          # ex: 78
print(result["quality_report"])         # dict complet
print(result["quality_report_json"])    # "/.../agent_8/output/quality_report.json"
print(result["quality_report_pdf"])     # "/.../agent_8/output/quality_report.pdf"
```

---

## 🔗 Intégration avec les autres agents

Agent 8 s'insère à la **fin du pipeline** et consomme les sorties de 3 agents :

```
Agent 1 (Ingestion)
      ↓
Agent 2 (Profiler) ─────────────────────────── dataset_profile ──────┐
      ↓                                                                │
Agent 3 (PII Detector) ─────────────────────── pii_flags ────────────┤
      ↓                                                                │
Agent 4 (Brain) → Agent 5 (AutoLabelEngine) ── labeled_records ──────┤
      ↓                                         intents, sentiments   │
Agent 6 (Cost Analyst)                                                 │
      ↓                                                                │
Agent 7 (Business Strategist)                                          │
      ↓                                                                │
Agent 8 (Quality Gate) ←──────────────────────────────────────────────┘
      ↓
  quality_report.json
  quality_report.pdf
  quality_score → intégrable dans l'état pour un Agent 9 futur
```

### Clés attendues dans le state (résumé)

| Clé | Produite par | Obligatoire |
|---|---|---|
| `labeled_records` | Agent 5 | Recommandé |
| `intents` | Agent 5 | Optionnel |
| `sentiments` | Agent 5 | Optionnel |
| `entities` | Agent 5 | Optionnel |
| `pii_flags` | Agent 3 | Optionnel (fallback regex) |
| `dataset_profile` | Agent 2 | Optionnel (fallback local) |
| `topic` | Utilisateur | Optionnel |

> **Note :** Aucune clé n'est strictement obligatoire. Agent 8 fonctionne en **dégradé gracieux** : si une donnée est absente, il utilise une méthode alternative (fallback). Le score sera moins précis mais l'agent ne crashera pas.

---

## 📦 Dépendances

Toutes les dépendances sont déjà listées dans `requirements.txt` à la racine du projet.

| Bibliothèque | Usage | Obligatoire |
|---|---|---|
| `reportlab` | Génération du rapport PDF | Non (PDF ignoré si absent) |
| `matplotlib` | Graphique en barres dans le PDF | Non (graphique ignoré si absent) |
| `pandas` | Manipulation de données (utils) | Oui |
| `scikit-learn` | Analyses statistiques | Oui |

### Installation rapide

```bash
pip install reportlab matplotlib
```

---

## 👥 Auteurs

Projet **NLP Dataset Intelligence Engine** – Pipeline multi-agents NLP.  
Branche : `Aicha` | Repository : [imanekadri/NLP-Dataset-Intelligence-Engine](https://github.com/imanekadri/NLP-Dataset-Intelligence-Engine)
