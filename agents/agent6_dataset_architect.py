from sklearn.model_selection import train_test_split

from pipeline.state import NLPPipelineState
from pipeline.config import PipelineConfig


def run_dataset_architect(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 6 — Dataset Architect.
    Construit la structure train/val/test et formate en JSONL-ready records.
    """
    config = PipelineConfig()
    texts = state.get("texts", [])
    raw_docs = state.get("raw_docs", [])
    labels = state.get("labels", {})
    dataset_type = state.get("dataset_type", "general_text")
    ml_format = state.get("ml_format", "text_classification")
    languages_detected = state.get("languages_detected", {})

    if not texts:
        print("[Agent 6] No texts, skipping.")
        return {**state, "splits": {}, "formatted_dataset": []}

    print(f"[Agent 6] Building dataset structure for {len(texts)} documents...")

    # Construire les records formatés
    records = []
    doc_ids = []
    for i, text in enumerate(texts):
        doc_id = raw_docs[i]["doc_id"] if i < len(raw_docs) else f"text_{i}"
        doc_ids.append(doc_id)

        label_info = labels.get(doc_id, {})
        lang = languages_detected.get(doc_id, "unknown")

        record = {
            "doc_id": doc_id,
            "text": text,
            "label": label_info.get("category", "unknown"),
            "intent": label_info.get("intent", "unknown"),
            "sentiment": label_info.get("sentiment", "unknown"),
            "language": lang,
            "dataset_type": dataset_type,
            "ml_format": ml_format,
        }

        # Ajouter les entités si pertinent
        if ml_format == "token_classification":
            record["entities"] = label_info.get("entities", [])

        records.append(record)

    # Split train/val/test
    if len(records) < 3:
        train_ids = doc_ids
        val_ids = []
        test_ids = []
    else:
        train_ids, temp_ids = train_test_split(
            doc_ids,
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            random_state=config.RANDOM_STATE,
        )
        if len(temp_ids) >= 2:
            relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=relative_test,
                random_state=config.RANDOM_STATE,
            )
        else:
            val_ids = temp_ids
            test_ids = []

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    # Ajouter le split à chaque record
    for record in records:
        if record["doc_id"] in train_ids:
            record["split"] = "train"
        elif record["doc_id"] in val_ids:
            record["split"] = "val"
        else:
            record["split"] = "test"

    print(f"[Agent 6] Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    return {
        **state,
        "splits": splits,
        "formatted_dataset": records,
    }
