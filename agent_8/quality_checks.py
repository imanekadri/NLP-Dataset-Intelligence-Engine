"""
==========================================================================
agent_8/quality_checks.py
==========================================================================
RÔLE DE CE FICHIER :
    Ce fichier contient toutes les analyses de qualité effectuées par
    Agent 8 - Quality Gate. C'est le cœur du traitement.

ORGANISATION :
    Ce fichier définit 5 classes indépendantes, chacune responsable
    d'une analyse spécifique :

    1. ToxicityChecker      → Détecte les textes négatifs ou toxiques
    2. ClassBalanceChecker  → Vérifie si les classes sont bien réparties
    3. PIIChecker           → Identifie les données personnelles (emails, etc.)
    4. NoiseAndDupChecker   → Détecte les doublons et textes inutiles
    5. QualityScoreCalculator → Calcule le score final (0 à 100)

D'OÙ VIENNENT LES DONNÉES ?
    - records (list[dict])  : chaque enregistrement vient de l'Agent 5
                              (AutoLabelEngine). Il contient les champs :
                              "text", "intent", "sentiment", "topic_id", etc.
    - pii_flags             : résultat de l'Agent 3 (PII Detector).
                              Liste ou dict des entités sensibles trouvées.
    - dataset_profile (dict): profil statistique généré par l'Agent 2
                              (Profiler). Contient "avg_noise_ratio", etc.
    - topic (str)           : le sujet principal attendu du dataset,
                              fourni par l'utilisateur ou l'orchestrateur.
==========================================================================
"""

from __future__ import annotations

import re                           # Pour les expressions régulières (détection PII)
from collections import Counter     # Pour compter les occurrences de chaque classe
from typing import Any              # Type générique Python


# ==========================================================================
# SEUILS DE DÉCISION (paramètres configurables)
# ==========================================================================
# Ces constantes définissent à partir de quel niveau un problème est signalé.
# Vous pouvez les ajuster selon votre contexte métier.

TOXICITY_THRESHOLD = 0.15
# → Si plus de 15% des textes sont négatifs/toxiques : problème signalé.

IMBALANCE_RATIO_THRESHOLD = 3.0
# → Si la classe la plus fréquente est 3× plus grande que la plus petite : déséquilibre.
#   Exemple : 300 textes "cancel_order" vs 50 textes "refund" → ratio = 6 → déséquilibre.

PII_RATE_THRESHOLD = 0.05
# → Si plus de 5% des entrées contiennent des données personnelles : problème signalé.

DUPLICATION_RATE_THRESHOLD = 0.05
# → Si plus de 5% des textes sont des doublons exacts : problème signalé.

NOISE_RATE_THRESHOLD = 0.10
# → Si plus de 10% des textes sont bruités (trop courts, incohérents) : problème signalé.

# Liste de mots-clés considérés comme toxiques.
# Utilisée en FALLBACK si l'Agent 5 n'a pas fourni d'analyse de sentiment.
TOXIC_KEYWORDS = {
    "hate", "stupid", "idiot", "kill", "violence", "racist", "abuse",   # Anglais
    "haine", "nul", "imbécile", "tuer", "violence", "raciste", "insulte",  # Français
}


# ==========================================================================
# CLASSE 1 : ToxicityChecker
# ==========================================================================
class ToxicityChecker:
    """
    RÔLE : Mesure le taux de contenu négatif ou toxique dans le dataset,
           et détecte les textes qui ne correspondent pas au sujet attendu.

    STRATÉGIE DE DÉTECTION (par ordre de priorité) :
        1. Si l'Agent 5 a fourni un label de sentiment → on utilise ce label.
           Un sentiment "negative" (ou "négatif", "neg") = contenu problématique.
        2. Si pas de sentiment disponible → on recherche des mots-clés toxiques
           directement dans le texte brut (méthode de secours / fallback).

    DÉTECTION HORS-SUJET :
        Un enregistrement est considéré hors-sujet si :
        - Son topic_id contient "unknown", OU
        - Le topic du dataset (ex: "customer_support") n'apparaît pas dans son topic.
    """

    def run(self, records: list[dict], topic: str) -> dict:
        """
        Lance l'analyse de toxicité sur tous les enregistrements.

        Paramètres :
            records (list[dict]) : Liste des enregistrements labellisés par Agent 5.
                                   Chaque entrée a les clés : "text", "sentiment",
                                   "topic_id" (ou "topic").
            topic   (str)        : Le sujet principal attendu du dataset
                                   (ex: "customer_support", "medical", etc.)

        Retourne (dict) :
            {
                "toxicity_rate"   : float, proportion de textes toxiques (0.0 à 1.0)
                "off_topic_rate"  : float, proportion de textes hors-sujet
                "toxic_count"     : int,   nombre absolu de textes toxiques
                "off_topic_count" : int,   nombre absolu de textes hors-sujet
                "issues"          : list[str], liste des problèmes détectés
            }
        """
        # Cas particulier : dataset vide → aucune analyse possible
        if not records:
            return {"toxicity_rate": 0.0, "off_topic_rate": 0.0, "issues": []}

        total = len(records)      # Nombre total d'enregistrements
        toxic_count = 0           # Compteur de textes toxiques
        off_topic_count = 0       # Compteur de textes hors-sujet

        # Parcours de chaque enregistrement du dataset
        for rec in records:
            sentiment = rec.get("sentiment", {})

            # ── DÉTECTION DE TOXICITÉ ────────────────────────────────────
            # Cas 1 : l'Agent 5 a fourni le sentiment sous forme de dict
            #         Exemple : {"label": "negative", "score": 0.87}
            if isinstance(sentiment, dict):
                label = sentiment.get("label", "").lower()
                if label in ("negative", "négatif", "neg"):
                    toxic_count += 1

            # Cas 2 : l'Agent 5 a fourni le sentiment directement comme chaîne
            #         Exemple : "negative"
            elif isinstance(sentiment, str) and sentiment.lower() in ("negative", "neg"):
                toxic_count += 1

            # Cas 3 (fallback) : pas de sentiment disponible
            #         On cherche des mots-clés toxiques dans le texte brut.
            else:
                text = rec.get("text", "").lower()
                if any(kw in text for kw in TOXIC_KEYWORDS):
                    toxic_count += 1

            # ── DÉTECTION HORS-SUJET ─────────────────────────────────────
            # On récupère le topic de l'enregistrement (champ "topic_id" ou "topic")
            rec_topic = str(rec.get("topic_id", rec.get("topic", ""))).lower()
            # Un texte est hors-sujet si son topic est inconnu ou ne correspond pas
            if "unknown" in rec_topic or (topic and topic.lower() not in rec_topic):
                off_topic_count += 1

        # Calcul des taux (valeurs entre 0 et 1)
        toxicity_rate = round(toxic_count / total, 4)
        off_topic_rate = round(off_topic_count / total, 4)

        # Génération de la liste des problèmes dépassant les seuils
        issues = []
        if toxicity_rate > TOXICITY_THRESHOLD:
            issues.append("toxic content")
        if off_topic_rate > 0.2:   # Seuil : > 20% de textes hors-sujet
            issues.append("off-topic content")

        return {
            "toxicity_rate": toxicity_rate,
            "off_topic_rate": off_topic_rate,
            "toxic_count": toxic_count,
            "off_topic_count": off_topic_count,
            "issues": issues,
        }


# ==========================================================================
# CLASSE 2 : ClassBalanceChecker
# ==========================================================================
class ClassBalanceChecker:
    """
    RÔLE : Vérifie si les classes (intents et sentiments) sont équilibrées
           dans le dataset.

    POURQUOI C'EST IMPORTANT ?
        Un dataset déséquilibré (ex: 90% "cancel_order" et 10% "refund") peut
        biaiser un modèle de ML. Il sera bon pour prédire "cancel_order" mais
        mauvais pour "refund".

    MÉTHODE DE DÉTECTION :
        On calcule le ratio entre la classe la plus fréquente et la moins fréquente.
        Si ce ratio dépasse IMBALANCE_RATIO_THRESHOLD (= 3.0), c'est déséquilibré.
        Exemple :
            intents = ["cancel", "cancel", "cancel", "refund"]
            distribution = {"cancel": 3, "refund": 1}
            ratio = 3 / 1 = 3.0  → LIMITE atteinte → signalé comme déséquilibré

    SOURCE DES DONNÉES :
        - Les intents et sentiments sont fournis par l'Agent 5 (AutoLabelEngine).
        - Si les listes sont vides, on les extrait directement depuis les records.
    """

    def run(self, records: list[dict], intents: list[str], sentiments: list[str]) -> dict:
        """
        Analyse la distribution des intents et des sentiments.

        Paramètres :
            records    (list[dict]) : Enregistrements labellisés (Agent 5).
            intents    (list[str])  : Liste des intents (ex: ["cancel", "refund", ...]).
            sentiments (list[str])  : Liste des sentiments (ex: ["positive", "negative"]).

        Retourne (dict) :
            {
                "intent_distribution"       : dict, ex: {"cancel": 150, "refund": 50}
                "sentiment_distribution"    : dict, ex: {"positive": 80, "negative": 20}
                "intent_imbalance_ratio"    : float, ex: 3.0
                "sentiment_imbalance_ratio" : float
                "class_balance"             : str, "balanced" ou "imbalanced"
                "is_imbalanced"             : bool
            }
        """
        # ── EXTRACTION DES INTENTS depuis les records si la liste est vide ──
        # Cela arrive si l'orchestrateur passe les records mais pas les listes séparées.
        if not intents:
            intents = [
                rec.get("intent", rec.get("classification", "unknown"))
                for rec in records
                if rec.get("intent") or rec.get("classification")
            ]

        # ── EXTRACTION DES SENTIMENTS depuis les records si la liste est vide ──
        if not sentiments:
            sentiments = []
            for rec in records:
                s = rec.get("sentiment", {})
                if isinstance(s, dict):
                    # Sentiment fourni sous forme de dict (ex: {"label": "positive"})
                    sentiments.append(s.get("label", "unknown"))
                elif isinstance(s, str):
                    # Sentiment fourni directement comme chaîne (ex: "positive")
                    sentiments.append(s)

        # Comptage des occurrences de chaque intent et sentiment
        intent_dist = dict(Counter(intents))
        sentiment_dist = dict(Counter(sentiments))

        # ── CALCUL DU RATIO DE DÉSÉQUILIBRE ─────────────────────────────
        def imbalance_ratio(dist: dict) -> float:
            """
            Calcule le ratio max/min d'une distribution de classes.
            Un ratio de 1.0 signifie parfaitement équilibré.
            Plus le ratio est élevé, plus le déséquilibre est fort.
            """
            if not dist:
                return 1.0   # Distribution vide → considérée équilibrée
            counts = [v for v in dist.values() if isinstance(v, (int, float))]
            if not counts or min(counts) == 0:
                return float("inf")   # Division par zéro → déséquilibre infini
            return round(max(counts) / min(counts), 2)

        intent_ratio = imbalance_ratio(intent_dist)
        sentiment_ratio = imbalance_ratio(sentiment_dist)

        # Le dataset est déséquilibré si l'un des deux ratios dépasse le seuil
        is_imbalanced = (
            intent_ratio > IMBALANCE_RATIO_THRESHOLD
            or sentiment_ratio > IMBALANCE_RATIO_THRESHOLD
        )

        return {
            "intent_distribution": intent_dist,
            "sentiment_distribution": sentiment_dist,
            "intent_imbalance_ratio": intent_ratio,
            "sentiment_imbalance_ratio": sentiment_ratio,
            "class_balance": "imbalanced" if is_imbalanced else "balanced",
            "is_imbalanced": is_imbalanced,
            # Note : pas de champ "issues" ici, c'est build_json_report qui le gère
        }


# ==========================================================================
# CLASSE 3 : PIIChecker
# ==========================================================================
class PIIChecker:
    """
    RÔLE : Analyse la présence de données personnelles sensibles (PII =
           Personally Identifiable Information) dans le dataset.

    TYPES DE PII RECHERCHÉS :
        - Adresses email (ex: jean.dupont@gmail.com)
        - Numéros de téléphone (ex: +33 6 12 34 56 78)
        - Numéros de carte bancaire (ex: 4539 1488 0343 6467)

    SOURCE DES DONNÉES :
        - En priorité, on utilise pii_flags fourni par l'Agent 3
          (PII Detector spécialisé, basé sur Microsoft Presidio).
        - En fallback (si Agent 3 n'a pas tourné), on applique des
          expressions régulières (regex) directement sur les textes bruts.

    FORMATS SUPPORTÉS pour pii_flags (Agent 3) :
        Format liste : [{"text": "...", "pii": [{"type": "EMAIL", ...}]}, ...]
        Format dict  : {"pii_count": 12, "types": ["EMAIL", "PHONE"]}
    """

    # ── EXPRESSIONS RÉGULIÈRES pour la détection locale (fallback) ──────
    # Ces patterns sont utilisés si l'Agent 3 n'est pas disponible.

    # Détecte les emails : ex: nom.prenom@domaine.com
    EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    # Détecte les numéros de téléphone : ex: +33 6 12 34 56 78, (01)23456789
    PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")

    # Détecte les numéros de carte bancaire : 13 à 16 chiffres consécutifs
    CC_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")

    def run(self, pii_flags: Any, records: list[dict]) -> dict:
        """
        Analyse les données PII à partir de la sortie de l'Agent 3.

        Paramètres :
            pii_flags (list | dict | None) : Sortie de l'Agent 3.
                                             None si l'Agent 3 n'a pas tourné.
            records   (list[dict])         : Enregistrements bruts (fallback).

        Retourne (dict) :
            {
                "pii_detected" : bool,  True si au moins une PII trouvée
                "pii_count"    : int,   nombre d'entrées avec PII
                "pii_rate"     : float, proportion d'entrées avec PII (0.0 à 1.0)
                "pii_types"    : list[str], types de PII trouvés (ex: ["EMAIL", "PHONE"])
                "issues"       : list[str], problèmes signalés si pii_rate > seuil
            }
        """
        pii_count = 0
        pii_types: list[str] = []
        total = max(len(records), 1)   # Évite la division par zéro

        # ── CAS 1 : pii_flags est une LISTE de dicts (format Agent 3 standard) ──
        # Chaque élément correspond à un document du dataset.
        # Exemple : [{"text": "Bonjour", "pii": []}, {"text": "email@x.com", "pii": [{"type": "EMAIL"}]}]
        if isinstance(pii_flags, list):
            for entry in pii_flags:
                if isinstance(entry, dict):
                    # Récupère la liste d'entités PII de cet enregistrement
                    entities = entry.get("pii", entry.get("entities", []))
                    if entities:
                        pii_count += 1   # Cet enregistrement contient au moins une PII
                        for e in entities:
                            ptype = e.get("type", e.get("entity_type", "UNKNOWN"))
                            if ptype not in pii_types:
                                pii_types.append(ptype)   # Collecte les types uniques
            total = max(len(pii_flags), 1)

        # ── CAS 2 : pii_flags est un DICT global (résumé) ──────────────────
        # Exemple : {"pii_count": 15, "types": ["EMAIL", "PHONE"], "total_pii": 15}
        elif isinstance(pii_flags, dict):
            # Essaie plusieurs clés possibles selon la version d'Agent 3
            pii_count = pii_flags.get("pii_count", pii_flags.get("total_pii", 0))
            pii_types = pii_flags.get("types", pii_flags.get("entity_types", []))

        # ── CAS 3 (FALLBACK) : Agent 3 absent → détection locale par regex ──
        # On parcourt tous les textes bruts et on y cherche des patterns PII.
        else:
            for rec in records:
                text = rec.get("text", "")
                if (
                    self.EMAIL_RE.search(text)   # Contient un email ?
                    or self.PHONE_RE.search(text)  # Contient un téléphone ?
                    or self.CC_RE.search(text)     # Contient un numéro de carte ?
                ):
                    pii_count += 1
                    # On regroupe sous un seul label générique pour le fallback
                    pii_types = list(set(pii_types + ["EMAIL/PHONE/CARD"]))

        # Calcul du taux de PII (proportion d'enregistrements concernés)
        pii_rate = round(pii_count / total, 4)
        pii_detected = pii_count > 0   # True dès qu'au moins une PII est trouvée

        issues = []
        if pii_rate > PII_RATE_THRESHOLD:
            issues.append("sensitive PII data detected")

        return {
            "pii_detected": pii_detected,
            "pii_count": pii_count,
            "pii_rate": pii_rate,
            "pii_types": pii_types,
            "issues": issues,
        }


# ==========================================================================
# CLASSE 4 : NoiseAndDupChecker
# ==========================================================================
class NoiseAndDupChecker:
    """
    RÔLE : Détecte deux types de problèmes qui dégradent la qualité du dataset :
           1. LES DOUBLONS    → textes identiques répétés plusieurs fois.
           2. LE BRUIT        → textes vides, trop courts ou sans sens.

    POURQUOI C'EST IMPORTANT ?
        - Les doublons biaisent l'entraînement (le modèle voit trop certains exemples).
        - Le bruit introduit du "bruit" : textes parasites, sans information utile.

    MÉTHODE DE DÉTECTION DES DOUBLONS :
        On normalise chaque texte (minuscules + suppression des espaces multiples)
        puis on parcourt la liste en gardant une mémoire des textes déjà vus.
        Si un texte est déjà dans la mémoire → c'est un doublon.

    MÉTHODE DE DÉTECTION DU BRUIT :
        1. Priorité → on utilise avg_noise_ratio de l'Agent 2 (Profiler).
           L'Agent 2 calcule déjà ce ratio sur tout le dataset.
        2. Fallback → on applique la méthode _is_noisy() sur chaque texte.

    CRITÈRES DE BRUIT (_is_noisy) :
        - Texte vide ou composé uniquement d'espaces.
        - Texte avec moins de 5 mots (trop court pour être utile).
        - Texte sans aucune lettre alphabétique (ex: "12345 !!!??").
    """

    MIN_TOKENS = 5   # Seuil minimum de mots pour qu'un texte soit considéré valide

    def run(self, records: list[dict], dataset_profile: dict) -> dict:
        """
        Analyse les doublons et le bruit dans le dataset.

        Paramètres :
            records         (list[dict]) : Enregistrements labellisés (Agent 5).
            dataset_profile (dict)       : Profil du dataset généré par Agent 2.
                                           Utilisé pour récupérer avg_noise_ratio.

        Retourne (dict) :
            {
                "duplication_rate" : float, proportion de doublons (ex: 0.08 = 8%)
                "noise_rate"       : float, proportion de textes bruités
                "duplicate_count"  : int,   nombre absolu de doublons
                "noisy_count"      : int,   nombre absolu de textes bruités
                "issues"           : list[str], problèmes dépassant les seuils
            }
        """
        # Dataset vide → retour immédiat avec des valeurs nulles
        if not records:
            return {
                "duplication_rate": 0.0,
                "noise_rate": 0.0,
                "duplicate_count": 0,
                "noisy_count": 0,
                "issues": [],
            }

        total = len(records)
        # Extraction de tous les textes bruts (on accepte "" si "text" absent)
        texts = [rec.get("text", "") for rec in records]

        # ── DÉTECTION DES DOUBLONS ───────────────────────────────────────
        # Normalisation : on met tout en minuscules et on supprime les espaces multiples.
        # Cela permet de détecter les doublons même avec des casses ou espaces différents.
        # Exemple : "Bonjour !" == "bonjour !"  après normalisation.
        normalized = [" ".join(t.lower().split()) for t in texts]

        seen: set[str] = set()   # Ensemble des textes déjà rencontrés
        duplicate_count = 0

        for t in normalized:
            if t in seen:
                # Ce texte a déjà été vu → c'est un doublon
                duplicate_count += 1
            else:
                # Premier fois qu'on voit ce texte → on l'ajoute à la mémoire
                seen.add(t)

        # ── DÉTECTION DU BRUIT ───────────────────────────────────────────
        # PRIORITÉ : utiliser le ratio de bruit calculé par l'Agent 2
        avg_noise_from_profile = dataset_profile.get("avg_noise_ratio", None)

        if avg_noise_from_profile is not None:
            # L'Agent 2 a déjà analysé le bruit → on utilise directement son résultat
            noisy_count = round(avg_noise_from_profile * total)
            noise_rate = round(avg_noise_from_profile, 4)
        else:
            # FALLBACK : Agent 2 non disponible → on applique notre propre détection
            noisy_count = 0
            for text in texts:
                if self._is_noisy(text):
                    noisy_count += 1
            noise_rate = round(noisy_count / total, 4)

        duplication_rate = round(duplicate_count / total, 4)

        # Génération des problèmes si les seuils sont dépassés
        issues = []
        if duplication_rate > DUPLICATION_RATE_THRESHOLD:
            issues.append("high duplication rate")
        if noise_rate > NOISE_RATE_THRESHOLD:
            issues.append("noisy data detected")

        return {
            "duplication_rate": duplication_rate,
            "noise_rate": noise_rate,
            "duplicate_count": duplicate_count,
            "noisy_count": noisy_count,
            "issues": issues,
        }

    def _is_noisy(self, text: str) -> bool:
        """
        Détermine si un texte individuel est considéré comme du bruit.

        Un texte est du bruit si :
            - Il est vide ou composé uniquement d'espaces.
            - Il a moins de MIN_TOKENS mots (trop court, peu informatif).
            - Il ne contient aucune lettre alphabétique (latin, arabe, etc.).

        Paramètres :
            text (str) : Le texte à évaluer.

        Retourne :
            True  → le texte est du bruit.
            False → le texte est potentiellement valide.
        """
        # Texte vide ou uniquement des espaces
        if not text or not text.strip():
            return True

        # Texte trop court (moins de MIN_TOKENS mots)
        tokens = text.split()
        if len(tokens) < self.MIN_TOKENS:
            return True

        # Texte sans aucune lettre : caractères latins (À-ÿ) ou arabes (ء-ي)
        # Exemple de bruit : "1234 !!!! 5678" → aucune lettre → bruit
        if not re.search(r"[a-zA-ZÀ-ÿ\u0600-\u06FF]", text):
            return True

        return False   # Le texte semble valide


# ==========================================================================
# CLASSE 5 : QualityScoreCalculator
# ==========================================================================
class QualityScoreCalculator:
    """
    RÔLE : Calcule le score global de qualité du dataset, de 0 (mauvais) à 100 (excellent).

    PRINCIPE DU CALCUL (système de pénalités) :
        On part d'un score parfait de 100, et on soustrait des points
        pour chaque problème détecté par les 4 analyzes précédentes.
        Le score final est : max(0, 100 - somme_des_pénalités)

    TABLE DES PÉNALITÉS :
        ┌─────────────────────────────────┬──────────────┬────────────────────────────┐
        │ Problème                        │ Pénalité max │ Calcul                     │
        ├─────────────────────────────────┼──────────────┼────────────────────────────┤
        │ Toxicité élevée                 │ -20 pts      │ toxicity_rate × 100       │
        │ Contenu hors-sujet              │ -10 pts      │ off_topic_rate × 50        │
        │ Déséquilibre de classes         │ -15 pts      │ fixe si is_imbalanced=True │
        │ Données PII sensibles           │ -15 pts      │ pii_rate × 100             │
        │ Taux de doublons élevé          │ -15 pts      │ dup_rate × 100             │
        │ Bruit dans les données          │ -15 pts      │ noise_rate × 80            │
        └─────────────────────────────────┴──────────────┴────────────────────────────┘

    EXEMPLE CONCRET :
        toxicity_rate = 0.20  → pénalité = min(0.20 × 100, 20) = 20 pts
        is_imbalanced = True  → pénalité = 15 pts
        Autres = 0
        Score final = 100 - 20 - 15 = 65 / 100
    """

    def compute(
        self,
        toxicity_result: dict,
        balance_result: dict,
        pii_result: dict,
        noise_result: dict,
    ) -> int:
        """
        Calcule le score de qualité en appliquant les pénalités.

        Paramètres :
            toxicity_result (dict) : Résultat de ToxicityChecker.run()
            balance_result  (dict) : Résultat de ClassBalanceChecker.run()
            pii_result      (dict) : Résultat de PIIChecker.run()
            noise_result    (dict) : Résultat de NoiseAndDupChecker.run()

        Retourne :
            int : Score entre 0 et 100.
        """
        score = 100.0   # On démarre avec un score parfait

        # ── PÉNALITÉ 1 : Toxicité ────────────────────────────────────────
        # Plus le taux de toxicité est élevé, plus la pénalité est forte.
        # Maximum : -20 points (atteint si toxicity_rate >= 20%).
        tox_rate = toxicity_result.get("toxicity_rate", 0.0)
        score -= min(tox_rate * 100, 20)

        # ── PÉNALITÉ 2 : Contenu hors-sujet ─────────────────────────────
        # Multiplicateur plus faible (×50) car le hors-sujet est moins critique.
        # Maximum : -10 points (atteint si off_topic_rate >= 20%).
        off_rate = toxicity_result.get("off_topic_rate", 0.0)
        score -= min(off_rate * 50, 10)

        # ── PÉNALITÉ 3 : Déséquilibre de classes ────────────────────────
        # Pénalité fixe de 15 points si un déséquilibre est détecté.
        if balance_result.get("is_imbalanced", False):
            score -= 15

        # ── PÉNALITÉ 4 : Données PII ─────────────────────────────────────
        # Maximum : -15 points (atteint si pii_rate >= 15%).
        pii_rate = pii_result.get("pii_rate", 0.0)
        score -= min(pii_rate * 100, 15)

        # ── PÉNALITÉ 5 : Doublons ────────────────────────────────────────
        # Maximum : -15 points (atteint si duplication_rate >= 15%).
        dup_rate = noise_result.get("duplication_rate", 0.0)
        score -= min(dup_rate * 100, 15)

        # ── PÉNALITÉ 6 : Bruit ───────────────────────────────────────────
        # Multiplicateur ×80 car le bruit est légèrement moins pénalisant que les PII.
        # Maximum : -15 points (atteint si noise_rate >= ~19%).
        noise_rate = noise_result.get("noise_rate", 0.0)
        score -= min(noise_rate * 80, 15)

        # Le score ne peut jamais être négatif (plancher à 0)
        return max(0, int(round(score)))
