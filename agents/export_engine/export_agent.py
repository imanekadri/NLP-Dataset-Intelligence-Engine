import os

from .exporters.jsonl_exporter import export_jsonl
from .exporters.csv_exporter import export_csv
from .exporters.hf_exporter import export_huggingface
from .exporters.openai_exporter import export_openai
from .exporters.rag_exporter import export_rag


class ExportEngineAgent:

    def __init__(self, export_root="exports"):
        self.export_root = export_root
        os.makedirs(export_root, exist_ok=True)

    def run(self, state):

        dataset = state.get("structured_dataset")
        dataset_type = state.get("dataset_type")
        ml_format = state.get("ml_format")

        print("Agent 7 — Export Engine started")

        results = {}

        results["jsonl"] = export_jsonl(dataset, self.export_root)
        results["csv"] = export_csv(dataset, self.export_root)
        results["huggingface"] = export_huggingface(dataset, self.export_root)

        # special formats
        if ml_format == "instruction_tuning":
            results["openai"] = export_openai(dataset, self.export_root)

        if dataset_type == "rag":
            results["rag"] = export_rag(dataset, self.export_root)

        state["exports"] = {
            "formats": results,
            "num_samples": len(dataset),
            "export_root": self.export_root
        }

        print("Agent 7 — Export completed")

        return state