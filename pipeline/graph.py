from langgraph.graph import StateGraph, START, END

from pipeline.state import NLPPipelineState
from agents.agent1_ingestion import run_ingestion
from agents.agent2_profiler import run_profiler
from agents.agent3_understanding import run_understanding
from agents.agent4_semantic_brain import run_semantic_brain
from agents.agent5_auto_label import run_auto_label
from agents.agent6_dataset_architect import run_dataset_architect
from agents.agent7_export_engine import run_export_engine
from agents.agent8_qa_agent import run_qa_agent
from agents.agent9_readme_generator import run_readme_generator


def build_pipeline():
    """Construit le pipeline LangGraph avec les 9 agents."""
    graph = StateGraph(NLPPipelineState)

    # Ajouter les nœuds
    graph.add_node("ingestion", run_ingestion)
    graph.add_node("profiler", run_profiler)
    graph.add_node("understanding", run_understanding)
    graph.add_node("semantic_brain", run_semantic_brain)
    graph.add_node("auto_label", run_auto_label)
    graph.add_node("dataset_architect", run_dataset_architect)
    graph.add_node("export_engine", run_export_engine)
    graph.add_node("qa_agent", run_qa_agent)
    graph.add_node("readme_generator", run_readme_generator)

    # Pipeline séquentiel
    graph.add_edge(START, "ingestion")
    graph.add_edge("ingestion", "profiler")
    graph.add_edge("profiler", "understanding")
    graph.add_edge("understanding", "semantic_brain")
    graph.add_edge("semantic_brain", "auto_label")
    graph.add_edge("auto_label", "dataset_architect")
    graph.add_edge("dataset_architect", "export_engine")
    graph.add_edge("export_engine", "qa_agent")
    graph.add_edge("qa_agent", "readme_generator")
    graph.add_edge("readme_generator", END)

    return graph.compile()
