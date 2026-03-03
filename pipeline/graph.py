from langgraph.graph import StateGraph
from agents.ingestion_agent1 import run_ingestion
from agents.profiler_agent2 import run_agent2
from agents.semantic_brain.agent import DatasetBrainAgent

brain_agent = DatasetBrainAgent()

def build_graph():

    graph = StateGraph(dict)

    graph.add_node("ingestion", run_ingestion)
    graph.add_node("agent2", run_agent2)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "agent2")

    return graph.compile()