from langgraph.graph import StateGraph, END
from agent45.agents.semantic_brain.agent import DatasetBrainAgent
from graph.state import GraphState
from core.logger import logger

brain_agent = DatasetBrainAgent()
def semantic_brain_node(state: GraphState) -> GraphState:
    try:
        decision = brain_agent.run(
            metadata=state["metadata"],
            sample_texts=state["sample_texts"]
        )

        return {
            **state,
            "brain_decision": decision,
            "enriched_metadata": {
                **state["metadata"]
            }
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [str(e)]
        }

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("semantic_brain", semantic_brain_node)

    builder.set_entry_point("semantic_brain")

    builder.add_edge("semantic_brain", END)

    return builder.compile()
