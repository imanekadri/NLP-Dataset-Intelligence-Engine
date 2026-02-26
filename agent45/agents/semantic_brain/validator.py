import json
from pydantic import ValidationError
from .schema import DatasetBrainOutput

def validate_output(raw_output: str) -> DatasetBrainOutput:
    try:
        parsed = json.loads(raw_output)
        validated = DatasetBrainOutput(**parsed)
        return validated
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid LLM output: {e}")
