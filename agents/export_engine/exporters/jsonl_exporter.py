import json
import os


def export_jsonl(dataset, root):

    if not dataset:
        return None

    path = os.path.join(root, "dataset.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            try:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[JSONL ERROR] Skipping row: {e}")

    return path