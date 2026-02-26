import json

SYSTEM_PROMPT = """You are a dataset architect.
Analyze the provided metadata including:
- topics
- entities
- structure
- language

Decide:
1. dataset type (e.g., Classification, NER, Chatbot, QA, Resume, Traduction, RAG, Instruction tuning, Code dataset)
2. best ML usage (tasks)
3. optimal structure (format)
4. labeling strategy (languages)

Respond ONLY with a valid JSON object matching this schema:
{
  "dataset_type": "string",
  "tasks": ["string", "string"],
  "format": "string",
  "languages": ["string", "string"]
}"""

def build_prompt(enriched_metadata: dict) -> str:
    """Formats the metadata into a readable string for the LLM."""
    metadata_json = json.dumps(enriched_metadata, indent=2, ensure_ascii=False)
    return f"Here is the dataset metadata to analyze:\n\n{metadata_json}"