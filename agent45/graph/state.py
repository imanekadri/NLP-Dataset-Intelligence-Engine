from typing import TypedDict, Dict, Any, List

class GraphState(TypedDict):
    metadata: Dict[str, Any]
    sample_texts: List[str]
    enriched_metadata: Dict[str, Any]
    brain_decision: Dict[str, Any]
    errors: List[str]
