from pipeline.state import NLPPipelineState


def run_auto_label(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 5 — Auto Label Engine.
    Lit NER, sentiment, topics depuis le state (PAS de recalcul).
    Génère un label composite par document.
    """
    texts = state.get("texts", [])
    raw_docs = state.get("raw_docs", [])
    ner_results = state.get("ner_results", {})
    sentiment_results = state.get("sentiment_results", {})
    topic_clusters = state.get("topic_clusters", {})
    dataset_type = state.get("dataset_type", "general_text")
    labeling_strategy = state.get("labeling_strategy", "topic_based")

    if not texts:
        print("[Agent 5] No texts, skipping.")
        return {**state, "labels": {}}

    print(f"[Agent 5] Labeling {len(texts)} documents (strategy: {labeling_strategy})...")

    labels = {}
    for i, text in enumerate(texts):
        doc_id = raw_docs[i]["doc_id"] if i < len(raw_docs) else f"text_{i}"

        # Récupérer les résultats existants depuis le state
        entities = ner_results.get(doc_id, [])
        sentiment = sentiment_results.get(doc_id, "unknown")

        # Extraire la catégorie principale des entités
        entity_labels = [e["label"] for e in entities] if entities else []
        primary_entity_type = max(set(entity_labels), key=entity_labels.count) if entity_labels else "NONE"

        # Intent basé sur la structure du texte
        if "?" in text[:200]:
            intent = "question"
        elif any(w in text[:100].lower() for w in ["please", "can you", "could you", "help"]):
            intent = "request"
        elif any(w in text[:100].lower() for w in ["thank", "great", "good"]):
            intent = "positive_feedback"
        else:
            intent = "statement"

        # Catégorie basée sur le type de dataset
        if dataset_type == "ner_dataset":
            category = primary_entity_type
        elif dataset_type == "sentiment_analysis":
            category = sentiment
        elif dataset_type == "conversational":
            category = intent
        else:
            category = "general"

        labels[doc_id] = {
            "intent": intent,
            "sentiment": sentiment,
            "entities": entities,
            "primary_entity_type": primary_entity_type,
            "category": category,
        }

    # Stats
    categories = [l["category"] for l in labels.values()]
    unique_categories = set(categories)
    print(f"[Agent 5] {len(labels)} documents labeled, {len(unique_categories)} unique categories.")

    return {**state, "labels": labels}
