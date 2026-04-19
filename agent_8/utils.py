"""
==========================================================================
agent_8/utils.py
==========================================================================
RÔLE DE CE FICHIER :
    Ce fichier contient des fonctions utilitaires partagées par tous les
    autres modules d'Agent 8. Il joue le rôle de "boîte à outils" commune.

RESPONSABILITÉS PRINCIPALES :
    1. Définir les chemins vers les fichiers produits par les agents précédents
       (Agent 2, Agent 3, Agent 5).
    2. Charger et sauvegarder des fichiers JSON de manière sécurisée.
    3. Fournir des fonctions d'affichage (logging) propres.

COMMENT CE FICHIER EST UTILISÉ :
    - main.py  importe load_json, DEFAULT_PATHS, OUTPUT_DIR, log_section
    - report_generator.py importe save_json, OUTPUT_DIR
==========================================================================
"""

import json                       # Pour lire/écrire des fichiers JSON
import os                         # Pour manipuler les chemins de fichiers
from datetime import datetime     # Pour générer l'horodatage
from typing import Any            # Type générique Python


# ==========================================================================
# CHEMINS PAR DÉFAUT DES SORTIES DES AGENTS PRÉCÉDENTS
# ==========================================================================
# BASE_DIR pointe vers la racine du projet (le dossier parent de agent_8/).
# On construit ensuite les chemins vers les fichiers JSON de chaque agent.
# Ces chemins sont utilisés en mode autonome : si Agent 8 est exécuté seul
# (sans orchestrateur), il va chercher les données des autres agents ici.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PATHS = {
    # Agent 2 (Profiler) → génère un profil statistique du dataset
    # Contient : avg_noise_ratio, dialogue_ratio, topics_detected, etc.
    "dataset_profile": os.path.join(BASE_DIR, "output", "agent2", "dataset_profile.json"),

    # Agent 3 (PII Detector) → détecte les données personnelles sensibles
    # Contient : liste d'entités PII (emails, numéros de téléphone, etc.)
    "pii_flags": os.path.join(BASE_DIR, "output", "agent3", "pii_report.json"),

    # Agent 5 (AutoLabelEngine) → labellise chaque texte du dataset
    # Contient : intent, sentiment, entities, topic_id pour chaque enregistrement
    "agent5_labels": os.path.join(BASE_DIR, "output", "agent5", "labels.json"),
}

# Dossier de sortie d'Agent 8 lui-même (quality_report.json + quality_report.pdf)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ==========================================================================
# FONCTION : load_json
# ==========================================================================
def load_json(path: str) -> Any:
    """
    Charge et retourne le contenu d'un fichier JSON.

    POURQUOI : Agent 8 doit lire les données produites par les agents 2, 3 et 5.
               Cette fonction gère les deux cas d'erreur courants :
               - Le fichier n'existe pas encore (agent précédent pas encore exécuté)
               - Le fichier existe mais est corrompu (JSON invalide)

    Paramètres :
        path (str) : Chemin absolu vers le fichier JSON à charger.

    Retourne :
        Any   : Le contenu Python (dict, list, etc.) si succès.
        None  : Si le fichier est introuvable ou invalide.
    """
    # Vérification de l'existence du fichier avant d'essayer de l'ouvrir
    if not os.path.exists(path):
        print(f"  [AVERTISSEMENT] Fichier introuvable : {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)       # Lecture et décodage du JSON en objet Python
    except json.JSONDecodeError as e:
        # Le fichier existe mais son contenu JSON est invalide (syntaxe incorrecte)
        print(f"  [ERREUR] JSON invalide dans {path} : {e}")
        return None


# ==========================================================================
# FONCTION : save_json
# ==========================================================================
def save_json(data: Any, path: str) -> None:
    """
    Sauvegarde un objet Python dans un fichier JSON indenté (lisible).

    POURQUOI : Agent 8 produit quality_report.json. Cette fonction crée
               automatiquement les dossiers parents si nécessaire, puis
               écrit le contenu de façon lisible (indent=2) et en UTF-8
               pour supporter les caractères accentués et arabes.

    Paramètres :
        data (Any) : L'objet Python à sérialiser (dict, list, etc.).
        path (str) : Chemin absolu du fichier de destination.
    """
    # Crée le dossier parent s'il n'existe pas (ex: agent_8/output/)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # ensure_ascii=False → conserve les accents et caractères non-ASCII
        # indent=2 → format indenté, facile à lire pour un humain
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  ✅ JSON sauvegardé : {path}")


# ==========================================================================
# FONCTION : timestamp
# ==========================================================================
def timestamp() -> str:
    """
    Retourne l'horodatage actuel au format ISO 8601.

    Exemple de retour : "2026-04-19T11:30:00.123456"

    POURQUOI : Chaque rapport généré par Agent 8 est horodaté pour traçabilité.
    """
    return datetime.now().isoformat()


# ==========================================================================
# FONCTION : log_section
# ==========================================================================
def log_section(title: str) -> None:
    """
    Affiche un séparateur visuel dans la console pour délimiter les sections.

    POURQUOI : Rendre l'exécution d'Agent 8 lisible dans le terminal,
               en séparant clairement les 4 analyses et le score final.

    Exemple d'affichage :
        ────────────────────────────────────────────────────────────
          🧪 Analyse 1 : Toxicité & Hors-sujet
        ────────────────────────────────────────────────────────────

    Paramètres :
        title (str) : Le titre de la section à afficher.
    """
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ==========================================================================
# FONCTION : clamp
# ==========================================================================
def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    Borne (limite) une valeur numérique entre min_val et max_val.

    POURQUOI : Garantit que le score de qualité reste toujours entre 0 et 100,
               même si les pénalités dépassent 100 points.

    Exemples :
        clamp(120)  →  100
        clamp(-5)   →  0
        clamp(75)   →  75

    Paramètres :
        value   (float) : La valeur à borner.
        min_val (float) : Borne inférieure (défaut : 0.0).
        max_val (float) : Borne supérieure (défaut : 100.0).
    """
    return max(min_val, min(max_val, value))
