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

    clusters: Dict[str, List[str]]   
    categories: str  
    trace_with_topics_csv: str   


    # Agent3

    agent3_results: List[Dict]  