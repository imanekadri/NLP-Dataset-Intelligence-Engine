import json
import os


def export_rag(dataset, root):

    if not dataset:
        return None

    path = os.path.join(root, "rag_documents.json")

    docs = []

    for i, row in enumerate(dataset):

        text = row.get("text", "")

        if not text:
            continue

        docs.append({
            "id": str(i),
            "text": text,
            "metadata": {
                k: v for k, v in row.items() if k != "text"
            }
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    return path