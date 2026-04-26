
# schema.py:
# from typing import List, Optional
# from pydantic import BaseModel
#
#
# class DatasetBrainOutput(BaseModel):
#     dataset_type: str
#     tasks: List[str]
#     ideal_schema: dict
#     optimal_format: str


# import json
# from pydantic import ValidationError
# from .schema import DatasetBrainOutput
# def validate_output(raw_output: str) -> DatasetBrainOutput:
#     try:
#         parsed = json.loads(raw_output)
#         validated = DatasetBrainOutput(**parsed)
#         return validated
#     except (json.JSONDecodeError, ValidationError) as e:
#         raise ValueError(f"Invalid LLM output: {e}")

import json
import re
from pydantic import ValidationError
from .schema import DatasetBrainOutput


def extract_json(text: str) -> str:
    # Extract JSON object from text using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return match.group(0)


def validate_output(raw_output: str) -> DatasetBrainOutput:
    try:
        json_str = extract_json(raw_output)
        parsed = json.loads(json_str)
        validated = DatasetBrainOutput(**parsed)
        return validated

    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid LLM structured output: {e}\nRaw output: {raw_output}")