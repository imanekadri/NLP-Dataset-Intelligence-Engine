import os
import sys
import json
from datetime import datetime

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.graph import build_pipeline
from pipeline.config import PipelineConfig


def main():
    config = PipelineConfig()
    config.ensure_dirs()

    print("=" * 60)
    print("  NLP Dataset Intelligence Engine — LangGraph Pipeline")
    print("=" * 60)
    print(f"  Input:  {config.DATASETS_DIR}")
    print(f"  Output: {config.OUTPUT_DIR}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pipeline = build_pipeline()

    initial_state = {
        "input_dir": config.DATASETS_DIR,
    }

    final_state = pipeline.invoke(initial_state)

    # Résumé final
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Documents processed: {len(final_state.get('raw_docs', []))}")
    print(f"  Duplicates removed:  {final_state.get('duplicates_removed', 0)}")
    print(f"  Dataset type:        {final_state.get('dataset_type', 'unknown')}")
    print(f"  ML format:           {final_state.get('ml_format', 'unknown')}")
    print(f"  Quality score:       {final_state.get('quality_score', 'N/A')}/100")
    print(f"  Export formats:      {len(final_state.get('export_paths', {}))}")
    print("=" * 60)

    # Sauvegarder le state final
    state_path = os.path.join(config.REPORTS_DIR, "final_state_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_docs": len(final_state.get("raw_docs", [])),
        "duplicates_removed": final_state.get("duplicates_removed", 0),
        "dataset_type": final_state.get("dataset_type", "unknown"),
        "ml_format": final_state.get("ml_format", "unknown"),
        "tasks": final_state.get("tasks", []),
        "quality_score": final_state.get("quality_score", 0),
        "export_paths": final_state.get("export_paths", {}),
        "dataset_profile": final_state.get("dataset_profile", {}),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {state_path}")


if __name__ == "__main__":
    main()
