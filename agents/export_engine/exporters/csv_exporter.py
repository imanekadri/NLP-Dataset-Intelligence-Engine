import csv
import os
import json


def export_csv(dataset, root):

    if not dataset:
        return None

    path = os.path.join(root, "dataset.csv")

    keys = dataset[0].keys()

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in dataset:
            safe_row = {}

            for k, v in row.items():
                if isinstance(v, (list, dict)):
                    safe_row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    safe_row[k] = v

            writer.writerow(safe_row)

    return path