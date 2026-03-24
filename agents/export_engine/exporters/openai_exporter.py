import json
import os


def export_openai(dataset, root):

    if not dataset:
        return None

    path = os.path.join(root, "openai_finetune.jsonl")

    with open(path, "w", encoding="utf-8") as f:

        for row in dataset:

            instruction = row.get("instruction", "")
            input_text = row.get("input", row.get("text", ""))
            output_text = row.get("output", row.get("intent", ""))

            example = {
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text}
                ]
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return path