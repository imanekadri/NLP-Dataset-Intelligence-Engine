from typing import TypedDict, List, Dict

class PipelineState(TypedDict, total=False):
    # Agent1
    raw_docs: List[Dict]
    extracted_files: List[str]

    languages_detected: Dict[str, str]
    duplicates_removed: int
    trace_csv_path: str
    report_json_path: str
    

    # Agent2
    dataset_profile: Dict
    clusters: Dict
    categories: str  

    # Agent3
    # agent3_results: List[Dict]  