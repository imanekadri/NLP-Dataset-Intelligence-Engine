"""
==========================================================================
agent_8/main.py
==========================================================================
RÔLE DE CE FICHIER :
    C'est le point d'entrée principal d'Agent 8 - Quality Gate.
    Il orchestre l'exécution des 4 analyses de qualité dans l'ordre,
    collecte leurs résultats, calcule le score global, puis génère
    les rapports de sortie (JSON et PDF).

DEUX MODES D'UTILISATION :

    Mode Pipeline (recommandé) :
    ─────────────────────────────
        Agent 8 reçoit un dictionnaire "state" de l'orchestrateur,
        contenant les données produites par les agents précédents.
        Il retourne ce même dictionnaire enrichi avec le rapport qualité.

        Exemple :
            from agent_8 import run_agent8
            result_state = run_agent8(state)

    Mode Autonome (test / debug) :
    ────────────────────────────────
        Agent 8 est lancé directement depuis le terminal.
        Il charge automatiquement les données depuis les fichiers JSON
        produits par les agents précédents (output/agent2/, etc.).

        Commande :
            python -X utf8 main.py

FLUX D'EXÉCUTION :
    1. Récupération des inputs (depuis state ou depuis les fichiers)
    2. Analyse 1 : Toxicité & hors-sujet         → ToxicityChecker
    3. Analyse 2 : Équilibre des classes          → ClassBalanceChecker
    4. Analyse 3 : Données personnelles (PII)     → PIIChecker
    5. Analyse 4 : Bruit & doublons               → NoiseAndDupChecker
    6. Calcul du score global (0–100)             → QualityScoreCalculator
    7. Génération des rapports JSON + PDF         → report_generator
    8. Retour de l'état enrichi au pipeline
==========================================================================
"""

from __future__ import annotations

import os
import sys

# ── CORRECTION ENCODAGE WINDOWS ─────────────────────────────────────────────
# Sous Windows, le terminal utilise l'encodage "cp1252" par défaut,
# qui ne supporte pas les caractères Unicode (emojis, accents arabes, etc.).
# On force l'encodage UTF-8 pour que les emojis et accents s'affichent correctement.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── PATH PYTHON ──────────────────────────────────────────────────────────────
# Ajoute le dossier agent_8/ au chemin de recherche Python.
# Cela permet d'importer utils, quality_checks et report_generator
# directement, sans avoir besoin de préfixe "agent_8.".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── IMPORTS DES MODULES D'AGENT 8 ────────────────────────────────────────────
from utils import (
    DEFAULT_PATHS,      # Chemins vers les fichiers JSON des agents précédents
    OUTPUT_DIR,         # Dossier de sortie d'Agent 8
    clamp,              # Fonction utilitaire pour borner une valeur
    load_json,          # Chargement sécurisé d'un fichier JSON
    log_section,        # Affichage d'un séparateur visuel dans le terminal
    timestamp,          # Génération d'un horodatage ISO
)
from quality_checks import (
    ClassBalanceChecker,       # Vérifie l'équilibre des classes (intents/sentiments)
    NoiseAndDupChecker,        # Détecte les doublons et le bruit
    PIIChecker,                # Détecte les données personnelles
    QualityScoreCalculator,    # Calcule le score global 0–100
    ToxicityChecker,           # Détecte le contenu négatif/toxique
)
from report_generator import (
    build_json_report,     # Assemble le rapport en dict Python
    save_json_report,      # Sauvegarde quality_report.json
    save_pdf_report,       # Génère quality_report.pdf
)


# ==========================================================================
# FONCTION PRINCIPALE : run_agent8
# ==========================================================================

def run_agent8(state: dict) -> dict:
    """
    Fonction principale d'Agent 8 – Quality Gate.

    RÔLE :
        Orchestrer toutes les analyses de qualité et produire un rapport
        complet avec un score et des recommandations d'amélioration.

    INPUTS ATTENDUS DANS state (dict) :
        "intents"          (list[str])    : Labels d'intention par Agent 5.
                                            Ex: ["cancel_order", "refund", ...]
        "entities"         (list)         : Entités nommées détectées par Agent 5.
                                            Ex: [{"text": "Paris", "type": "LOC"}]
        "sentiments"       (list[str])    : Sentiments par Agent 5.
                                            Ex: ["positive", "negative", "neutral"]
        "labeled_records"  (list[dict])   : Enregistrements complets labellisés.
                                            C'est la liste des dicts avec "text",
                                            "intent", "sentiment", "topic_id", etc.
        "pii_flags"        (list|dict)    : Sortie de l'Agent 3 (PII Detector).
                                            Peut être None si Agent 3 n'a pas tourné.
        "dataset_profile"  (dict)         : Profil statistique de l'Agent 2.
                                            Contient "avg_noise_ratio", "topics_detected", etc.
        "topic"            (str)          : Sujet principal du dataset.
                                            Ex: "customer_support", "medical"

    OUTPUTS RETOURNÉS (dict enrichi) :
        "quality_report"      (dict) : Le rapport complet (JSON sérialisable).
        "quality_score"       (int)  : Score global entre 0 et 100.
        "quality_report_json" (str)  : Chemin absolu vers quality_report.json.
        "quality_report_pdf"  (str)  : Chemin absolu vers quality_report.pdf.
    """
    # Affichage de l'en-tête dans le terminal
    print("\n" + "═" * 60)
    print("  🚀 Agent 8 – Quality Gate  |  Démarrage")
    print("═" * 60)

    # Création du dossier de sortie (agent_8/output/) s'il n'existe pas encore
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── ÉTAPE 1 : RÉCUPÉRATION DES INPUTS DEPUIS L'ÉTAT ──────────────────
    # On extrait chaque donnée du dictionnaire d'état du pipeline.
    # Si une donnée est absente, on utilise une valeur par défaut vide.
    intents         = state.get("intents", [])
    entities        = state.get("entities", [])
    sentiments      = state.get("sentiments", [])
    labeled_records = state.get("labeled_records", [])
    pii_flags       = state.get("pii_flags", None)
    dataset_profile = state.get("dataset_profile", {})
    topic           = state.get("topic", "")

    # ── ÉTAPE 2 : CHARGEMENT AUTOMATIQUE DEPUIS LES FICHIERS ─────────────
    # Si certaines données sont absentes de l'état (mode autonome),
    # on essaie de les charger depuis les fichiers JSON des agents précédents.
    # Cette fonctionnalité permet d'exécuter Agent 8 indépendamment.
    _load_missing_inputs_from_files(
        state, labeled_records, pii_flags, dataset_profile
    )
    # On relit les variables après le chargement éventuel depuis les fichiers
    labeled_records = state.get("labeled_records", labeled_records)
    pii_flags       = state.get("pii_flags", pii_flags)
    dataset_profile = state.get("dataset_profile", dataset_profile)

    # ── ÉTAPE 3 : RECONSTRUCTION DES RECORDS SI NÉCESSAIRE ───────────────
    # Si labeled_records est vide mais qu'on a des listes d'intents/sentiments,
    # on reconstruit des pseudo-enregistrements pour permettre les analyses
    # de distribution des classes. C'est un mode dégradé mais fonctionnel.
    if not labeled_records and (intents or sentiments):
        labeled_records = _rebuild_records_from_lists(intents, sentiments)

    # Affichage d'un résumé des données reçues
    print(f"\n  📦 Enregistrements chargés : {len(labeled_records)}")
    print(f"  📌 Topic                   : {topic or '(non spécifié)'}")

    # ──────────────────────────────────────────────────────────────────────
    # ANALYSE 1 : TOXICITÉ & CONTENU HORS-SUJET
    # Source des données : labeled_records (Agent 5) + topic (utilisateur)
    # ──────────────────────────────────────────────────────────────────────
    log_section("🧪 Analyse 1 : Toxicité & Hors-sujet")
    toxicity_checker = ToxicityChecker()
    toxicity_result  = toxicity_checker.run(labeled_records, topic)
    print(f"  Taux toxicité  : {toxicity_result['toxicity_rate']:.2%}")
    print(f"  Taux hors-sujet: {toxicity_result['off_topic_rate']:.2%}")
    print(f"  Problèmes      : {toxicity_result['issues'] or 'aucun'}")

    # ──────────────────────────────────────────────────────────────────────
    # ANALYSE 2 : ÉQUILIBRE DES CLASSES
    # Source des données : intents et sentiments (Agent 5)
    # ──────────────────────────────────────────────────────────────────────
    log_section("⚖️  Analyse 2 : Équilibre des Classes")
    balance_checker = ClassBalanceChecker()
    # On passe les listes brutes ET les records pour que le checker puisse
    # extraire les données depuis les records si les listes sont vides.
    balance_result  = balance_checker.run(labeled_records, intents, sentiments)
    print(f"  Statut         : {balance_result['class_balance']}")
    print(f"  Ratio intents  : {balance_result['intent_imbalance_ratio']}×")
    print(f"  Ratio sentiments : {balance_result['sentiment_imbalance_ratio']}×")

    # ──────────────────────────────────────────────────────────────────────
    # ANALYSE 3 : DONNÉES PERSONNELLES (PII)
    # Source des données : pii_flags (Agent 3) en priorité, sinon regex fallback
    # ──────────────────────────────────────────────────────────────────────
    log_section("🔐 Analyse 3 : Données Personnelles (PII)")
    pii_checker  = PIIChecker()
    pii_result   = pii_checker.run(pii_flags, labeled_records)
    print(f"  PII détectée   : {'Oui' if pii_result['pii_detected'] else 'Non'}")
    print(f"  Taux PII       : {pii_result['pii_rate']:.2%}")
    print(f"  Types PII      : {pii_result['pii_types'] or 'N/A'}")
    print(f"  Problèmes      : {pii_result['issues'] or 'aucun'}")

    # ──────────────────────────────────────────────────────────────────────
    # ANALYSE 4 : BRUIT & DOUBLONS
    # Source des données : labeled_records (Agent 5) + dataset_profile (Agent 2)
    # ──────────────────────────────────────────────────────────────────────
    log_section("🔁 Analyse 4 : Bruit & Doublons")
    noise_checker = NoiseAndDupChecker()
    # dataset_profile est utilisé pour récupérer avg_noise_ratio de l'Agent 2
    noise_result  = noise_checker.run(labeled_records, dataset_profile)
    print(f"  Taux doublons  : {noise_result['duplication_rate']:.2%}")
    print(f"  Taux bruit     : {noise_result['noise_rate']:.2%}")
    print(f"  Problèmes      : {noise_result['issues'] or 'aucun'}")

    # ──────────────────────────────────────────────────────────────────────
    # CALCUL DU SCORE GLOBAL (0–100)
    # On passe les 4 résultats d'analyse au calculateur de score.
    # Chaque problème détecté réduit le score selon une table de pénalités.
    # ──────────────────────────────────────────────────────────────────────
    log_section("🏆 Calcul du Score Global")
    calculator    = QualityScoreCalculator()
    quality_score = calculator.compute(
        toxicity_result, balance_result, pii_result, noise_result
    )
    print(f"\n  ⭐ Quality Score : {quality_score} / 100")

    # ──────────────────────────────────────────────────────────────────────
    # GÉNÉRATION DES RAPPORTS
    # 1. build_json_report() assemble toutes les métriques en un seul dict
    # 2. save_json_report()  écrit quality_report.json dans agent_8/output/
    # 3. save_pdf_report()   génère quality_report.pdf avec tableaux + graphiques
    # ──────────────────────────────────────────────────────────────────────
    log_section("📄 Génération des Rapports")

    # Assemblage du rapport final
    report = build_json_report(
        quality_score, toxicity_result, balance_result, pii_result, noise_result, topic
    )
    # Sauvegarde JSON
    json_path = save_json_report(report)
    # Sauvegarde PDF (retourne None si reportlab est absent)
    pdf_path  = save_pdf_report(report)

    # ── RÉSUMÉ FINAL DANS LE TERMINAL ────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  ✅ Agent 8 terminé | Score : {quality_score}/100")
    if report["issues"]:
        print(f"  ⚠️  Problèmes : {', '.join(report['issues'])}")
    else:
        print("  ✅ Aucun problème détecté !")
    print("═" * 60 + "\n")

    # ── RETOUR ENRICHI POUR LE PIPELINE ──────────────────────────────────
    # On retourne le state original + les nouvelles données d'Agent 8.
    # L'opérateur **state copie toutes les clés existantes du state (Agent 1-7),
    # puis on ajoute les clés spécifiques à Agent 8.
    return {
        **state,                              # Toutes les données des agents précédents
        "quality_report": report,             # Rapport complet (dict)
        "quality_score": quality_score,       # Score numérique 0–100
        "quality_report_json": json_path,     # Chemin vers quality_report.json
        "quality_report_pdf": pdf_path,       # Chemin vers quality_report.pdf
    }


# ==========================================================================
# FONCTIONS HELPER PRIVÉES (préfixe _ = usage interne uniquement)
# ==========================================================================

def _load_missing_inputs_from_files(
    state: dict,
    labeled_records: list,
    pii_flags,
    dataset_profile: dict,
) -> None:
    """
    Charge automatiquement les données manquantes depuis les fichiers JSON.

    POURQUOI CETTE FONCTION ?
        En mode autonome (sans orchestrateur), l'état (state) ne contient
        pas les données des agents précédents. Cette fonction va chercher
        ces données dans les fichiers de sortie des autres agents.
        Si les fichiers n'existent pas, un avertissement est affiché
        mais l'agent continue son exécution normalement.

    Paramètres (tous modifiés via "state" par référence) :
        state           (dict)       : Dictionnaire d'état partagé du pipeline.
        labeled_records (list)       : Enregistrements Agent 5 (vide si absent).
        pii_flags       (list|dict)  : Données Agent 3 (None si absent).
        dataset_profile (dict)       : Profil Agent 2 (vide si absent).
    """
    # ── Chargement du profil dataset de l'Agent 2 ───────────────────────
    # Le profil contient des statistiques utiles sur le dataset
    # (avg_noise_ratio, dialogue_ratio, topics_detected, etc.)
    if not dataset_profile:
        data = load_json(DEFAULT_PATHS["dataset_profile"])
        if data:
            state["dataset_profile"] = data
            print(f"  📂 dataset_profile chargé depuis fichier.")

    # ── Chargement des PII flags de l'Agent 3 ───────────────────────────
    # L'Agent 3 (PII Detector) produit un rapport sur les données personnelles trouvées.
    if pii_flags is None:
        data = load_json(DEFAULT_PATHS["pii_flags"])
        if data:
            state["pii_flags"] = data
            print(f"  📂 pii_flags chargé depuis fichier.")

    # ── Chargement des enregistrements labellisés de l'Agent 5 ──────────
    # L'Agent 5 (AutoLabelEngine) produit un fichier avec tous les enregistrements
    # labellisés (intent, sentiment, entities, topic_id pour chaque texte).
    if not labeled_records:
        data = load_json(DEFAULT_PATHS["agent5_labels"])

        if isinstance(data, list):
            # Format standard : la racine du JSON est directement une liste
            state["labeled_records"] = data
            print(f"  📂 labeled_records chargé depuis fichier ({len(data)} entrées).")

        elif isinstance(data, dict):
            # Certaines versions d'Agent 5 encapsulent la liste dans un dict
            # Ex: {"results": [...]} ou {"data": [...]}
            for key in ("results", "data", "records", "labels"):
                if key in data and isinstance(data[key], list):
                    state["labeled_records"] = data[key]
                    print(f"  📂 labeled_records chargé depuis fichier (clé '{key}').")
                    break


def _rebuild_records_from_lists(
    intents: list[str], sentiments: list[str]
) -> list[dict]:
    """
    Reconstruit une liste minimale d'enregistrements depuis les listes intents/sentiments.

    POURQUOI CETTE FONCTION ?
        Parfois, l'orchestrateur passe les intents et sentiments sous forme
        de simples listes (ex: ["cancel", "refund"]) plutôt que de passer
        la liste complète des records. Cette fonction crée des pseudo-records
        pour que les analyses de ClassBalanceChecker puissent fonctionner.

    Exemple :
        intents    = ["cancel", "refund"]
        sentiments = ["negative", "positive"]
        → retourne [
            {"text": "[record_0]", "intent": "cancel",  "sentiment": {"label": "negative"}},
            {"text": "[record_1]", "intent": "refund",  "sentiment": {"label": "positive"}},
          ]

    Paramètres :
        intents    (list[str]) : Liste des intents fournis par Agent 5.
        sentiments (list[str]) : Liste des sentiments fournis par Agent 5.

    Retourne :
        list[dict] : Liste de pseudo-enregistrements structurés.
    """
    records = []
    # On prend le max des deux longueurs pour ne perdre aucune donnée
    max_len = max(len(intents), len(sentiments))

    for i in range(max_len):
        # Chaque record a a minima un champ "text" (placeholder ici)
        rec: dict = {"text": f"[record_{i}]"}

        # Ajout de l'intent si disponible pour cet index
        if i < len(intents):
            rec["intent"] = intents[i]

        # Ajout du sentiment (sous forme de dict, format attendu par les checkers)
        if i < len(sentiments):
            rec["sentiment"] = {"label": sentiments[i]}

        records.append(rec)

    return records


# ==========================================================================
# POINT D'ENTRÉE EN MODE AUTONOME
# ==========================================================================
# Ce bloc s'exécute uniquement quand on lance le fichier directement :
#     python -X utf8 main.py
# Il ne s'exécute PAS quand le module est importé par un orchestrateur.

if __name__ == "__main__":
    print("🚀 Lancement d'Agent 8 en mode autonome...")

    # ── ÉTAT DE DÉMONSTRATION ─────────────────────────────────────────────
    # En mode autonome, on fournit un état minimal.
    # Agent 8 chargera automatiquement les données des autres agents
    # depuis les fichiers JSON dans output/agent2/, output/agent3/, output/agent5/.
    #
    # Pour tester avec des données réelles, remplacez les commentaires ci-dessous
    # par vos vraies données ou laissez vide pour le chargement automatique.
    demo_state = {
        "topic": "customer_support",    # Sujet du dataset à analyser
        # "intents":          [...],    # Optionnel : liste des intents Agent 5
        # "sentiments":       [...],    # Optionnel : liste des sentiments Agent 5
        # "labeled_records":  [...],    # Optionnel : enregistrements Agent 5
        # "pii_flags":        [...],    # Optionnel : données PII Agent 3
        # "dataset_profile":  {...},    # Optionnel : profil Agent 2
    }

    # Lancement de l'agent
    result_state = run_agent8(demo_state)

    # Affichage du résultat final
    print(f"\n📊 Quality Score final : {result_state['quality_score']} / 100")
    print(f"📄 Rapport JSON        : {result_state.get('quality_report_json', 'N/A')}")
    print(f"📄 Rapport PDF         : {result_state.get('quality_report_pdf', 'N/A')}")
