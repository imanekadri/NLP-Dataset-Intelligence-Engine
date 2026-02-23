import os
import json
import csv

from pipeline.state import NLPPipelineState
from pipeline.config import PipelineConfig


def run_export_engine(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 7 — Export Engine.
    Exporte le dataset formaté en JSONL, CSV, HuggingFace et OpenAI fine-tune format.
    """
    config = PipelineConfig()
    config.ensure_dirs()

    records = state.get("formatted_dataset", [])
    splits = state.get("splits", {})

    if not records:
        print("[Agent 7] No formatted dataset, skipping.")
        return {**state, "export_paths": {}}

    print(f"[Agent 7] Exporting {len(records)} records...")
    export_paths = {}

    # --- JSONL complet ---
    jsonl_path = os.path.join(config.EXPORTS_DIR, "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    export_paths["jsonl"] = jsonl_path

    # --- CSV ---
    csv_path = os.path.join(config.EXPORTS_DIR, "dataset.csv")
    csv_fields = ["doc_id", "text", "label", "intent", "sentiment", "language", "split"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    export_paths["csv"] = csv_path

    # --- HuggingFace format (split par fichier) ---
    hf_dir = os.path.join(config.EXPORTS_DIR, "huggingface")
    os.makedirs(hf_dir, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_records = [r for r in records if r.get("split") == split_name]
        if split_records:
            split_path = os.path.join(hf_dir, f"{split_name}.jsonl")
            with open(split_path, "w", encoding="utf-8") as f:
                for record in split_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            export_paths[f"hf_{split_name}"] = split_path

    # Dataset info pour HF
    dataset_info = {
        "dataset_type": state.get("dataset_type", "unknown"),
        "ml_format": state.get("ml_format", "unknown"),
        "total_records": len(records),
        "splits": {k: len(v) for k, v in splits.items()},
        "labels": list(set(r.get("label", "") for r in records)),
    }
    info_path = os.path.join(hf_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    export_paths["hf_info"] = info_path

    # --- OpenAI fine-tune format ---
    openai_path = os.path.join(config.EXPORTS_DIR, "openai_finetune.jsonl")
    with open(openai_path, "w", encoding="utf-8") as f:
        for record in records:
            openai_record = {
                "messages": [
                    {"role": "system", "content": f"You are a {state.get('dataset_type', 'text')} classifier."},
                    {"role": "user", "content": record["text"][:2000]},
                    {"role": "assistant", "content": json.dumps({
                        "label": record.get("label", ""),
                        "sentiment": record.get("sentiment", ""),
                        "intent": record.get("intent", ""),
                    })},
                ]
            }
            f.write(json.dumps(openai_record, ensure_ascii=False) + "\n")
    export_paths["openai_finetune"] = openai_path

    print(f"[Agent 7] Exported to {len(export_paths)} formats.")
    for fmt, path in export_paths.items():
        print(f"  - {fmt}: {path}")

    return {**state, "export_paths": export_paths}
