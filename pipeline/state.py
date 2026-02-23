from typing import TypedDict, List, Dict


class NLPPipelineState(TypedDict, total=False):
    # --- Input ---
    input_dir: str

    # --- Agent 1 : Data Ingestion ---
    raw_docs: List[Dict]
    trace_csv_path: str
    extracted_files: List[str]
    languages_detected: Dict[str, str]
    duplicates_removed: int

    # --- Agent 2 : Profiler NLP ---
    texts: List[str]
    topic_clusters: Dict
    dataset_profile: Dict

    # --- Agent 3 : NLP Understanding ---
    embeddings: Dict[str, List[float]]
    ner_results: Dict[str, List[Dict]]
    sentiment_results: Dict[str, str]
    pii_flags: Dict[str, bool]

    # --- Agent 4 : Semantic Brain ---
    dataset_type: str
    ml_format: str
    tasks: List[str]
    labeling_strategy: str

    # --- Agent 5 : Auto Label ---
    labels: Dict[str, Dict]

    # --- Agent 6 : Dataset Architect ---
    splits: Dict[str, List[str]]
    formatted_dataset: List[Dict]

    # --- Agent 7 : Export Engine ---
    export_paths: Dict[str, str]

    # --- Agent 8 : QA Agent ---
    quality_score: float
    qa_report: Dict

    # --- Agent 9 : README Generator ---
    readme_content: str
