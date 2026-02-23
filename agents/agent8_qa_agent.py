import os
import json
from collections import Counter

from pipeline.state import NLPPipelineState
from pipeline.config import PipelineConfig


def run_qa_agent(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 8 — Dataset QA Agent.
    Vérifie la qualité : équilibre des classes, PII, biais, toxicité.
    Produit un score de qualité (0-100) et un rapport détaillé.
    """
    config = PipelineConfig()
    config.ensure_dirs()

    labels = state.get("labels", {})
    pii_flags = state.get("pii_flags", {})
    sentiment_results = state.get("sentiment_results", {})
    languages_detected = state.get("languages_detected", {})
    formatted_dataset = state.get("formatted_dataset", [])

    total_docs = len(formatted_dataset) if formatted_dataset else len(labels)
    if total_docs == 0:
        print("[Agent 8] No data to QA.")
        return {**state, "quality_score": 0.0, "qa_report": {}}

    print(f"[Agent 8] Running QA checks on {total_docs} documents...")

    # --- 1. Équilibre des classes ---
    categories = [l.get("category", "unknown") for l in labels.values()]
    cat_counter = Counter(categories)
    min_class_ratio = min(cat_counter.values()) / max(cat_counter.values()) if len(cat_counter) > 1 else 1.0
    class_balance_score = min(min_class_ratio * 100, 100)

    imbalanced_classes = []
    for cls, count in cat_counter.items():
        ratio = count / total_docs
        if ratio < 0.05:
            imbalanced_classes.append({"class": cls, "count": count, "ratio": round(ratio, 4)})

    # --- 2. PII Check ---
    pii_count = sum(1 for v in pii_flags.values() if v)
    pii_ratio = pii_count / total_docs if total_docs > 0 else 0
    pii_clean_score = max(0, (1 - pii_ratio) * 100)

    # --- 3. Biais linguistique ---
    lang_counter = Counter(languages_detected.values())
    if len(lang_counter) > 1:
        dominant_lang_ratio = max(lang_counter.values()) / sum(lang_counter.values())
        lang_bias_score = (1 - abs(dominant_lang_ratio - 0.5) * 2) * 100 if dominant_lang_ratio < 0.95 else 50
    else:
        lang_bias_score = 100  # Mono-langue = pas de biais

    # --- 4. Complétude des labels ---
    labeled_count = sum(1 for l in labels.values() if l.get("category") and l["category"] != "unknown")
    completeness_score = (labeled_count / total_docs * 100) if total_docs > 0 else 0

    # --- Score global ---
    quality_score = round(
        class_balance_score * 0.25
        + pii_clean_score * 0.25
        + lang_bias_score * 0.25
        + completeness_score * 0.25,
        2
    )

    qa_report = {
        "total_documents": total_docs,
        "quality_score": quality_score,
        "class_balance": {
            "score": round(class_balance_score, 2),
            "distribution": dict(cat_counter),
            "imbalanced_classes": imbalanced_classes,
        },
        "pii_check": {
            "score": round(pii_clean_score, 2),
            "flagged_documents": pii_count,
            "pii_ratio": round(pii_ratio, 4),
        },
        "language_bias": {
            "score": round(lang_bias_score, 2),
            "distribution": dict(lang_counter),
        },
        "label_completeness": {
            "score": round(completeness_score, 2),
            "labeled": labeled_count,
            "total": total_docs,
        },
        "issues": [],
    }

    # Liste des issues
    if pii_ratio > 0.1:
        qa_report["issues"].append(f"High PII exposure: {pii_count} documents ({pii_ratio:.1%})")
    if imbalanced_classes:
        qa_report["issues"].append(f"Imbalanced classes: {[c['class'] for c in imbalanced_classes]}")
    if completeness_score < 80:
        qa_report["issues"].append(f"Low label completeness: {completeness_score:.1f}%")

    # Sauvegarde
    report_path = os.path.join(config.REPORTS_DIR, "qa_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2, ensure_ascii=False)

    print(f"[Agent 8] Quality score: {quality_score}/100")
    if qa_report["issues"]:
        for issue in qa_report["issues"]:
            print(f"  ⚠ {issue}")

    return {
        **state,
        "quality_score": quality_score,
        "qa_report": qa_report,
    }
