from typing import List, Optional
from pydantic import BaseModel

class DatasetBrainOutput(BaseModel):
    dataset_type: str
    tasks: List[str]
    format: str
    languages: List[str]

    domain: Optional[str] = None
    label_strategy: Optional[str] = None
    risk_flags: Optional[List[str]] = None
    confidence_score: Optional[float] = None