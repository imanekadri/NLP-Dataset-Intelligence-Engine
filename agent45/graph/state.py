from typing import TypedDict, Dict, Any, List

class GraphState(TypedDict):
    metadata: Dict[str, Any]
    sample_texts: List[str]
    enriched_metadata: Dict[str, Any]
    brain_decision: Dict[str, Any]
    errors: List[str]


# """
# core/state.py — Shared LangGraph pipeline state (memory between all agents)
# This is the single source of truth that flows through every agent node.
# """
#
# from typing import TypedDict, Optional, Any
# from dataclasses import dataclass, field
# from enum import Enum
#
#
# # ─────────────────────────────────────────────
# # ENUMS
# # ─────────────────────────────────────────────
#
# class DatasetType(str, Enum):
#     CLASSIFICATION      = "classification"
#     NER                 = "ner"
#     CHATBOT             = "chatbot_training"
#     QA                  = "qa"
#     SUMMARIZATION       = "summarization"
#     TRANSLATION         = "translation"
#     RAG                 = "rag"
#     INSTRUCTION_TUNING  = "instruction_tuning"
#     CODE                = "code_dataset"
#     UNKNOWN             = "unknown"
#
#
# class ExportFormat(str, Enum):
#     HUGGINGFACE     = "huggingface"
#     JSONL           = "jsonl"
#     CSV             = "csv"
#     RAG             = "rag"
#     INSTRUCTION     = "instruction_tuning"
#     OPENAI          = "openai"
#
#
# class PipelineStatus(str, Enum):
#     PENDING     = "pending"
#     RUNNING     = "running"
#     COMPLETED   = "completed"
#     FAILED      = "failed"
#     NEEDS_RETRY = "needs_retry"
#
#
# # ─────────────────────────────────────────────
# # DOCUMENT MODEL
# # ─────────────────────────────────────────────
#
# @dataclass
# class Document:
#     doc_id: str
#     text: str
#     source: str = ""
#     language: str = "unknown"
#     entities: list[dict] = field(default_factory=list)
#     topic: str = ""
#     sentiment: str = "neutral"
#     intent: str = ""
#     embedding: list[float] = field(default_factory=list)
#     label: str = ""
#     metadata: dict = field(default_factory=dict)
#     is_duplicate: bool = False
#     is_toxic: bool = False
#     has_pii: bool = False
#
#
# # ─────────────────────────────────────────────
# # PIPELINE STATE (LangGraph shared memory)
# # ─────────────────────────────────────────────
#
# class PipelineState(TypedDict):
#     # ── Input ──────────────────────────────────
#     input_paths: list[str]
#     pipeline_id: str
#
#     # ── Agent 1 — Ingestion ────────────────────
#     raw_documents: list[dict]
#     ingestion_report: dict
#
#     # ── Agent 2 — Profiler ─────────────────────
#     profile_report: dict
#     detected_languages: list[str]
#     detected_structure: str        # dialogue | paragraph | code | mixed
#
#     # ── Agent 3 — NLP Understanding ───────────
#     analyzed_documents: list[dict]
#     nlp_report: dict
#
#     # ── Agent 4 — Semantic Brain ───────────────
#     dataset_type: str
#     tasks: list[str]
#     export_format: str
#     labeling_strategy: dict
#     brain_decision: dict
#
#     # ── Agent 5 — Auto Label ───────────────────
#     labeled_documents: list[dict]
#     label_stats: dict
#
#     # ── Agent 6 — Dataset Architect ───────────
#     structured_dataset: dict
#     dataset_schema: dict
#
#     # ── Agent 7 — Export Engine ────────────────
#     exported_paths: dict
#     export_report: dict
#
#     # ── Agent 8 — QA ──────────────────────────
#     qa_score: float
#     qa_issues: list[str]
#     qa_report: dict
#     qa_passed: bool
#
#     # ── Agent 9 — README ──────────────────────
#     readme_content: str
#     documentation_path: str
#
#     # ── Orchestration ─────────────────────────
#     current_agent: str
#     errors: list[str]
#     retry_count: int
#     logs: list[str]
#     status: str