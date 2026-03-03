
from typing import List, Dict
from pydantic import BaseModel


class LabelingStrategy(BaseModel):
    type: str
    labels: List[str]


class DatasetBrainOutput(BaseModel):
    dataset_type: str
    tasks: List[str]
    ideal_schema: Dict[str, str]
    labeling_strategy: LabelingStrategy
    optimal_format: str


# class DatasetBrainOutput(BaseModel):
#     dataset_type: str
#     tasks: List[str]
#     ideal_schema :dict
#     optimal_format:str
#
#     domain: Optional[str] = None
#     label_strategy: Optional[dict] = None
#     risk_flags: Optional[List[str]] = None
#     confidence_score: Optional[float] = None
