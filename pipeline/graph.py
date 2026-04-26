from langgraph.graph import StateGraph
from agents.agent1_Ingestion.agent1_ingestion import run_ingestion
from agents.agent1_Ingestion.agent1_cleaning import run_cleaning
from agents.agent2_profiler import run_agent2
from agents.agent3_und import run_agent3
from agents.semantic_brain.agent import DatasetBrainAgent

# brain_agent = DatasetBrainAgent()

def build_graph():

    graph = StateGraph(dict)

    graph.add_node("agent1", run_ingestion)
    graph.add_node("cleaning", run_cleaning) 
    graph.add_node("agent2", run_agent2)
    graph.add_node("agent3", run_agent3)
    

    graph.set_entry_point("agent1")
    graph.add_edge("agent1", "cleaning")
    graph.add_edge("cleaning", "agent2")
    graph.add_edge("agent2", "agent3")

    return graph.compile()