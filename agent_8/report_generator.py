"""
==========================================================================
agent_8/report_generator.py
==========================================================================
RÔLE DE CE FICHIER :
    Ce fichier est responsable de la génération des deux rapports de sortie
    d'Agent 8 :
        1. quality_report.json → rapport structuré, exploitable par d'autres programmes
        2. quality_report.pdf  → rapport lisible par un humain, avec tableaux et graphiques

BIBLIOTHÈQUES UTILISÉES :
    - reportlab  : génération de documents PDF (tableaux, styles, mise en page).
    - matplotlib : création de graphiques en barres (intégré dans le PDF).
    Ces deux bibliothèques sont importées de manière optionnelle : si elles
    sont absentes, l'agent continue sans générer le PDF (pas de crash).

FONCTIONS PRINCIPALES :
    - build_json_report()        → assemble le rapport comme un dict Python
    - _generate_recommendations() → génère les conseils selon les problèmes trouvés
    - save_json_report()         → écrit le dict dans quality_report.json
    - save_pdf_report()          → construit et sauvegarde quality_report.pdf
    - _build_bar_chart()         → crée le graphique en barres (matplotlib)
    - _score_color()             → couleur de fond du score (vert/orange/rouge)
    - _default_table_style()     → style visuel des tableaux dans le PDF
==========================================================================
"""

from __future__ import annotations

import io                          # Pour créer un buffer en mémoire (graphique PNG)
import os
from datetime import datetime
from typing import Any

from utils import save_json, OUTPUT_DIR   # Fonctions utilitaires d'Agent 8

# ==========================================================================
# IMPORTS OPTIONNELS (dégradation gracieuse si bibliothèque absente)
# ==========================================================================
# Si reportlab n'est pas installé, le PDF ne sera pas généré,
# mais le JSON et toutes les analyses fonctionneront normalement.
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4                          # Format de page A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm                              # Unité de mesure en centimètres
    from reportlab.platypus import (
        Image,            # Pour intégrer des images (graphique matplotlib)
        Paragraph,        # Pour du texte stylisé
        SimpleDocTemplate,  # Structure principale du document PDF
        Spacer,           # Pour ajouter de l'espace vertical
        Table,            # Pour les tableaux
        TableStyle,       # Pour le style des tableaux
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("  [INFO] reportlab non disponible – le PDF ne sera pas généré.")

# Si matplotlib n'est pas installé, le graphique est ignoré (PDF généré sans graphique).
try:
    import matplotlib
    matplotlib.use("Agg")           # Mode non-interactif : pas besoin d'écran/fenêtre
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==========================================================================
# FONCTION : build_json_report
# ==========================================================================
def build_json_report(
    quality_score: int,
    toxicity_result: dict,
    balance_result: dict,
    pii_result: dict,
    noise_result: dict,
    topic: str,
) -> dict:
    """
    Assemble le rapport de qualité final sous forme de dictionnaire Python.

    POURQUOI CETTE FONCTION ?
        Elle centralise toutes les métriques calculées par les 4 checkers
        en un seul objet structuré. Ce dict est ensuite sauvegardé en JSON
        et utilisé pour générer le PDF.

    Paramètres :
        quality_score    (int)  : Score global 0–100 calculé par QualityScoreCalculator.
        toxicity_result  (dict) : Résultat de ToxicityChecker.
        balance_result   (dict) : Résultat de ClassBalanceChecker.
        pii_result       (dict) : Résultat de PIIChecker.
        noise_result     (dict) : Résultat de NoiseAndDupChecker.
        topic            (str)  : Sujet principal du dataset.

    Retourne (dict) : Le rapport complet, conforme au schéma de sortie attendu.
    """
    # ── COLLECTE DE TOUS LES PROBLÈMES DÉTECTÉS ─────────────────────────
    # On rassemble les "issues" de chaque checker dans une seule liste.
    all_issues: list[str] = (
        toxicity_result.get("issues", [])
        + balance_result.get("issues", []) if "issues" in balance_result
        else (["class imbalance"] if balance_result.get("is_imbalanced") else [])
        + pii_result.get("issues", [])
        + noise_result.get("issues", [])
    )

    # ── DÉDUPLICATION : on supprime les doublons en conservant l'ordre ───
    seen_issues: set[str] = set()
    unique_issues: list[str] = []
    for issue in all_issues:
        if issue not in seen_issues:
            seen_issues.add(issue)
            unique_issues.append(issue)

    # ── GÉNÉRATION DES RECOMMANDATIONS ───────────────────────────────────
    # Les recommandations sont générées automatiquement selon les issues.
    recommendations = _generate_recommendations(
        unique_issues, toxicity_result, balance_result, pii_result, noise_result
    )

    # ── ASSEMBLAGE DU RAPPORT FINAL ───────────────────────────────────────
    report = {
        "agent": "Agent8_QualityGate",          # Identifiant de cet agent
        "timestamp": datetime.now().isoformat(), # Date et heure d'exécution
        "topic": topic,                          # Sujet du dataset analysé
        "quality_score": quality_score,          # Score global (0–100)
        "toxicity_rate": toxicity_result.get("toxicity_rate", 0.0),
        "off_topic_rate": toxicity_result.get("off_topic_rate", 0.0),
        "class_balance": balance_result.get("class_balance", "unknown"),
        "intent_distribution": balance_result.get("intent_distribution", {}),
        "sentiment_distribution": balance_result.get("sentiment_distribution", {}),
        "pii_detected": pii_result.get("pii_detected", False),
        "pii_rate": pii_result.get("pii_rate", 0.0),
        "pii_types": pii_result.get("pii_types", []),
        "duplication_rate": noise_result.get("duplication_rate", 0.0),
        "noise_rate": noise_result.get("noise_rate", 0.0),
        "issues": unique_issues,                 # Liste des problèmes détectés
        "recommendations": recommendations,      # Conseils d'amélioration
    }
    return report


# ==========================================================================
# FONCTION PRIVÉE : _generate_recommendations
# ==========================================================================
def _generate_recommendations(
    issues: list[str],
    toxicity_result: dict,
    balance_result: dict,
    pii_result: dict,
    noise_result: dict,
) -> list[str]:
    """
    Génère automatiquement des recommandations d'amélioration personnalisées.

    LOGIQUE :
        Pour chaque type de problème détecté (issue), on génère un conseil
        concret qui inclut les chiffres (nombre de textes concernés, taux, etc.)
        pour que le message soit actionnable.

    Paramètres :
        issues           (list[str]) : Liste des problèmes détectés.
        toxicity_result  (dict) : Pour récupérer toxic_count, toxicity_rate.
        balance_result   (dict) : Pour récupérer intent_imbalance_ratio.
        pii_result       (dict) : Pour récupérer pii_types, pii_count.
        noise_result     (dict) : Pour récupérer duplicate_count, noisy_count.

    Retourne :
        list[str] : Liste de recommandations en français.
                    Si aucun problème → message positif retourné.
    """
    recs: list[str] = []

    # Recommandation si du contenu toxique a été détecté
    if "toxic content" in issues:
        rate = toxicity_result.get("toxicity_rate", 0)
        recs.append(
            f"Nettoyer ou supprimer les {toxicity_result.get('toxic_count', 0)} "
            f"textes toxiques/négatifs ({rate:.1%} du dataset)."
        )

    # Recommandation si des textes hors-sujet ont été détectés
    if "off-topic content" in issues:
        recs.append(
            f"Filtrer les {toxicity_result.get('off_topic_count', 0)} entrées "
            "hors-sujet détectées (topic 'unknown')."
        )

    # Recommandation si les classes sont déséquilibrées
    if balance_result.get("is_imbalanced"):
        ratio = balance_result.get("intent_imbalance_ratio", 0)
        recs.append(
            f"Rééquilibrer les classes (ratio déséquilibre : {ratio}×). "
            "Utiliser sur-échantillonnage (SMOTE) ou sous-échantillonnage."
        )

    # Recommandation si des données PII ont été détectées
    if "sensitive PII data detected" in issues:
        types = ", ".join(pii_result.get("pii_types", [])) or "inconnues"
        recs.append(
            f"Anonymiser ou supprimer les données PII ({types}) "
            f"présentes dans {pii_result.get('pii_count', 0)} entrées."
        )

    # Recommandation si le taux de doublons est trop élevé
    if "high duplication rate" in issues:
        recs.append(
            f"Dédupliquer le dataset : {noise_result.get('duplicate_count', 0)} "
            f"doublons détectés ({noise_result.get('duplication_rate', 0):.1%})."
        )

    # Recommandation si des textes bruités ont été détectés
    if "noisy data detected" in issues:
        recs.append(
            f"Supprimer ou corriger les {noise_result.get('noisy_count', 0)} "
            "textes bruités (trop courts, symboles, incohérents)."
        )

    # Si aucun problème n'a été détecté → message positif
    if not recs:
        recs.append("Le dataset est de bonne qualité. Aucune action corrective nécessaire.")

    return recs


# ==========================================================================
# FONCTION : save_json_report
# ==========================================================================
def save_json_report(report: dict) -> str:
    """
    Sauvegarde le rapport sous forme de fichier quality_report.json.

    Paramètres :
        report (dict) : Le rapport assemblé par build_json_report().

    Retourne :
        str : Le chemin absolu du fichier JSON sauvegardé.
    """
    path = os.path.join(OUTPUT_DIR, "quality_report.json")
    save_json(report, path)   # Utilise la fonction utilitaire de utils.py
    return path


# ==========================================================================
# FONCTION : save_pdf_report
# ==========================================================================
def save_pdf_report(report: dict) -> str | None:
    """
    Génère et sauvegarde le rapport sous forme de fichier quality_report.pdf.

    STRUCTURE DU PDF (de haut en bas) :
        1. En-tête  : titre, date, topic du dataset
        2. Tableau des métriques : score, taux de toxicité, PII, doublons...
        3. Graphique en barres   : visualisation des métriques (si matplotlib available)
        4. Distribution des intents   (si données disponibles)
        5. Distribution des sentiments (si données disponibles)
        6. Liste des problèmes détectés
        7. Recommandations numérotées
        8. Pied de page

    Paramètres :
        report (dict) : Le rapport complet (sortie de build_json_report).

    Retourne :
        str  : Chemin vers le PDF généré.
        None : Si reportlab n'est pas installé.
    """
    # Vérification de disponibilité de reportlab
    if not REPORTLAB_AVAILABLE:
        print("  [AVERTISSEMENT] Impossible de générer le PDF (reportlab absent).")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, "quality_report.pdf")

    # ── CRÉATION DU DOCUMENT PDF ─────────────────────────────────────────
    # SimpleDocTemplate gère automatiquement la pagination et les marges.
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # getSampleStyleSheet() retourne les styles de base de reportlab
    styles = getSampleStyleSheet()

    # "story" est la liste des éléments à placer dans le PDF (dans l'ordre)
    story: list[Any] = []

    # ── STYLES PERSONNALISÉS ──────────────────────────────────────────────
    # On crée des styles sur mesure pour avoir un rendu professionnel.
    title_style = ParagraphStyle(
        "Title8",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1a237e"),   # Bleu foncé
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "Heading8",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#283593"),   # Bleu indigo
        spaceBefore=14,
        spaceAfter=4,
    )
    normal_style = styles["Normal"]
    bullet_style = ParagraphStyle(
        "Bullet8",
        parent=styles["Normal"],
        leftIndent=20,     # Indentation pour les listes à puces
        bulletIndent=10,
        spaceBefore=2,
    )

    # ── EN-TÊTE ───────────────────────────────────────────────────────────
    story.append(Paragraph("Agent 8 - Quality Gate Report", title_style))
    story.append(Paragraph(f"Généré le : {report.get('timestamp', '')}", normal_style))
    story.append(Paragraph(f"Topic : <b>{report.get('topic', 'N/A')}</b>", normal_style))
    story.append(Spacer(1, 0.5 * cm))

    # ── TABLEAU DES MÉTRIQUES ─────────────────────────────────────────────
    # Affiche le score et toutes les métriques dans un tableau coloré.
    score = report.get("quality_score", 0)
    score_color = _score_color(score)   # Vert si bon score, rouge si mauvais
    story.append(Paragraph("Score Global de Qualité", heading_style))
    score_data = [
        ["Métrique", "Valeur"],
        ["Quality Score", f"{score} / 100"],
        ["Toxicity Rate", f"{report.get('toxicity_rate', 0):.2%}"],
        ["Off-Topic Rate", f"{report.get('off_topic_rate', 0):.2%}"],
        ["Class Balance", report.get("class_balance", "N/A")],
        ["PII Detected", "Oui" if report.get("pii_detected") else "Non"],
        ["PII Rate", f"{report.get('pii_rate', 0):.2%}"],
        ["Duplication Rate", f"{report.get('duplication_rate', 0):.2%}"],
        ["Noise Rate", f"{report.get('noise_rate', 0):.2%}"],
    ]
    score_table = Table(score_data, colWidths=[9 * cm, 7 * cm])
    score_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),  # En-tête bleu
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8eaf6"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fa8da")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            # La ligne du score est colorée selon la valeur (vert/orange/rouge)
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor(score_color)),
            ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, 1), 13),
        ])
    )
    story.append(score_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── GRAPHIQUE EN BARRES (optionnel) ───────────────────────────────────
    # Appelle _build_bar_chart() pour créer une image PNG en mémoire.
    chart_img = _build_bar_chart(report)
    if chart_img:
        story.append(Paragraph("Aperçu des Métriques", heading_style))
        story.append(chart_img)
        story.append(Spacer(1, 0.4 * cm))

    # ── DISTRIBUTION DES INTENTS ──────────────────────────────────────────
    intent_dist = report.get("intent_distribution", {})
    if intent_dist:
        story.append(Paragraph("Distribution des Intents", heading_style))
        rows = [["Intent", "Nombre"]] + [[k, str(v)] for k, v in intent_dist.items()]
        t = Table(rows, colWidths=[10 * cm, 5 * cm])
        t.setStyle(_default_table_style())
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))

    # ── DISTRIBUTION DES SENTIMENTS ───────────────────────────────────────
    sentiment_dist = report.get("sentiment_distribution", {})
    if sentiment_dist:
        story.append(Paragraph("Distribution des Sentiments", heading_style))
        rows = [["Sentiment", "Nombre"]] + [[k, str(v)] for k, v in sentiment_dist.items()]
        t = Table(rows, colWidths=[10 * cm, 5 * cm])
        t.setStyle(_default_table_style())
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))

    # ── PROBLÈMES DÉTECTÉS ────────────────────────────────────────────────
    issues = report.get("issues", [])
    story.append(Paragraph("Problèmes Détectés", heading_style))
    if issues:
        for issue in issues:
            story.append(Paragraph(f"• {issue}", bullet_style))
    else:
        story.append(Paragraph("Aucun problème majeur détecté.", normal_style))
    story.append(Spacer(1, 0.3 * cm))

    # ── RECOMMANDATIONS ───────────────────────────────────────────────────
    recommendations = report.get("recommendations", [])
    story.append(Paragraph("Recommandations", heading_style))
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", bullet_style))
    story.append(Spacer(1, 0.5 * cm))

    # ── PIED DE PAGE ──────────────────────────────────────────────────────
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
    )
    story.append(Paragraph(
        "Rapport généré automatiquement par le pipeline NLP Dataset Intelligence Engine – Agent 8.",
        footer_style,
    ))

    # Construction finale du PDF à partir de la liste "story"
    doc.build(story)
    print(f"  ✅ PDF sauvegardé : {pdf_path}")
    return pdf_path


# ==========================================================================
# FONCTIONS HELPER PRIVÉES
# ==========================================================================

def _score_color(score: int) -> str:
    """
    Retourne une couleur hexadécimale selon la valeur du score.

    Logique visuelle :
        >= 80 : Vert  → bon dataset
        >= 60 : Orange → dataset acceptable mais à améliorer
        < 60  : Rouge  → dataset de mauvaise qualité, action urgente

    Paramètres :
        score (int) : Score de qualité entre 0 et 100.

    Retourne :
        str : Code couleur hexadécimal (ex: "#2e7d32").
    """
    if score >= 80:
        return "#2e7d32"   # Vert foncé → bonne qualité
    elif score >= 60:
        return "#f57f17"   # Orange → qualité moyenne
    else:
        return "#b71c1c"   # Rouge foncé → mauvaise qualité


def _default_table_style() -> TableStyle:
    """
    Retourne un style de tableau standardisé pour les tableaux secondaires du PDF.

    Ce style est appliqué aux tableaux de distribution des intents et sentiments.
    Il utilise des bandes alternées (bleu clair / blanc) pour la lisibilité.

    Retourne :
        TableStyle : Objet de style reportlab prêt à l'emploi.
    """
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),  # En-tête indigo
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        # Bandes alternées : bleu très clair et blanc
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8eaf6"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fa8da")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ])


def _build_bar_chart(report: dict):
    """
    Génère un graphique en barres des métriques clés et le retourne
    comme objet Image intégrable dans le PDF reportlab.

    MÉTRIQUES AFFICHÉES :
        Toxicité, Hors-sujet, PII, Doublons, Bruit (toutes en pourcentage).

    COMMENT ÇA MARCHE ?
        1. matplotlib crée le graphique en mémoire (pas de fichier temporaire).
        2. On sauvegarde l'image dans un buffer BytesIO (en format PNG).
        3. reportlab lit ce buffer et crée un objet Image intégrable dans le PDF.

    Paramètres :
        report (dict) : Le rapport complet (sortie de build_json_report).

    Retourne :
        Image reportlab : Prêt à être ajouté dans story[].
        None            : Si matplotlib ou reportlab n'est pas disponible.
    """
    if not MATPLOTLIB_AVAILABLE or not REPORTLAB_AVAILABLE:
        return None

    # Données du graphique : labels et valeurs en pourcentage
    labels = ["Toxicité", "Hors-sujet", "PII", "Doublons", "Bruit"]
    values = [
        report.get("toxicity_rate", 0) * 100,       # Converti en %
        report.get("off_topic_rate", 0) * 100,
        report.get("pii_rate", 0) * 100,
        report.get("duplication_rate", 0) * 100,
        report.get("noise_rate", 0) * 100,
    ]
    # Une couleur distincte par métrique pour faciliter la lecture
    bar_colors = ["#e53935", "#fb8c00", "#8e24aa", "#1e88e5", "#43a047"]

    # Création du graphique matplotlib
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Taux (%)", fontsize=10)
    ax.set_title("Métriques de qualité (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(max(values) * 1.3, 10))

    # Ligne rouge pointillée indiquant le seuil critique à 10%
    ax.axhline(y=10, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Seuil critique (10%)")
    ax.legend(fontsize=8)

    # Annotations : affiche la valeur exacte au-dessus de chaque barre
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    # Sauvegarde en mémoire (buffer PNG) → pas de fichier temporaire sur le disque
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=120)
    plt.close(fig)   # Libère la mémoire matplotlib
    buf.seek(0)      # Remet le curseur au début du buffer pour la lecture

    # Création de l'objet Image reportlab à partir du buffer
    return Image(buf, width=14 * cm, height=6 * cm)
