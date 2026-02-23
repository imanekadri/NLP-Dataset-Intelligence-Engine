from pipeline.state import NLPPipelineState


def _infer_dataset_type(profile, ner_results, sentiment_results):
    """Heuristique pour inférer le type de dataset et le format ML optimal."""
    structure = profile.get("structure", "Paragraph")
    topics = profile.get("topics", {})
    total_docs = profile.get("statistical", {}).get("total_docs", 0)

    # Compter les entités NER
    total_entities = sum(len(v) for v in ner_results.values())
    avg_entities = total_entities / max(total_docs, 1)

    # Distribution du sentiment
    sentiments = list(sentiment_results.values())
    has_sentiment_variety = len(set(sentiments)) > 1

    # Inférence du type
    if structure == "Code":
        dataset_type = "code_generation"
        ml_format = "instruction_tuning"
        tasks = ["code_completion", "code_generation"]
    elif structure == "Dialogue":
        dataset_type = "conversational"
        ml_format = "instruction_tuning"
        tasks = ["intent_classification", "dialogue_generation"]
    elif avg_entities > 3:
        dataset_type = "ner_dataset"
        ml_format = "token_classification"
        tasks = ["named_entity_recognition", "information_extraction"]
    elif has_sentiment_variety:
        dataset_type = "sentiment_analysis"
        ml_format = "text_classification"
        tasks = ["sentiment_classification", "opinion_mining"]
    else:
        dataset_type = "general_text"
        ml_format = "text_classification"
        tasks = ["text_classification"]

    # Stratégie de labeling
    if dataset_type in ["ner_dataset"]:
        labeling_strategy = "entity_based"
    elif dataset_type in ["sentiment_analysis"]:
        labeling_strategy = "sentiment_based"
    elif dataset_type in ["conversational"]:
        labeling_strategy = "intent_based"
    else:
        labeling_strategy = "topic_based"

    return dataset_type, ml_format, tasks, labeling_strategy


def run_semantic_brain(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 4 — Semantic Brain.
    Analyse le profil, NER et sentiment pour inférer le type de dataset et le format ML.
    Mode heuristique par défaut, LLM optionnel si clé API disponible.
    """
    profile = state.get("dataset_profile", {})
    ner_results = state.get("ner_results", {})
    sentiment_results = state.get("sentiment_results", {})

    print("[Agent 4] Inferring dataset type and ML format...")

    dataset_type, ml_format, tasks, labeling_strategy = _infer_dataset_type(
        profile, ner_results, sentiment_results
    )

    print(f"[Agent 4] Dataset type: {dataset_type}, ML format: {ml_format}")
    print(f"[Agent 4] Tasks: {tasks}")
    print(f"[Agent 4] Labeling strategy: {labeling_strategy}")

    return {
        **state,
        "dataset_type": dataset_type,
        "ml_format": ml_format,
        "tasks": tasks,
        "labeling_strategy": labeling_strategy,
    }
