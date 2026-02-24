from typing import TypedDict, List, Dict

class PipelineState(TypedDict, total=False):
    raw_docs: List[Dict]
    ingestion_stats: Dict