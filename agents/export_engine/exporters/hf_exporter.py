from datasets import Dataset
import os


def export_huggingface(dataset, root):

    if not dataset:
        return None

    path = os.path.join(root, "hf_dataset")

    try:
        hf_dataset = Dataset.from_list(dataset)
        hf_dataset.save_to_disk(path)
    except Exception as e:
        print(f"[HF ERROR] {e}")
        return None

    return path