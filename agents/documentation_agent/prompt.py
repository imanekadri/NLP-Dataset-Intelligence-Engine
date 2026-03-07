SYSTEM_PROMPT = """
You are an AI documentation generator for NLP datasets.

Generate a professional dataset README in Markdown format.
"""


def build_prompt(data: dict) -> str:
    return f"""
Generate a complete dataset documentation.

Dataset information:
{data}

The README must include:

# NLP Dataset Report
## Dataset Type
## Tasks
## Languages
## Dataset Structure
## Classes
## Usage
## Training Example
## Quality Score

Return only the Markdown document.
"""