import os
import pandas as pd
import json
import ast
from core.config import agent1_config
from pipeline.graph import run_agent2, run_agent3
from langgraph.graph import StateGraph

def load_previous_results():
    """Loads agent1 results from the trace CSV."""
    csv_path = agent1_config.TRACE_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find previous results at {csv_path}. Please run the full pipeline once first.")

    print(f"Loading previous results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert dataframe back to list of dicts
    docs = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        
        # Repair key_words (stored as string representation of list or comma-separated)
        if isinstance(doc.get('key_words'), str):
            try:
                doc['key_words'] = ast.literal_eval(doc['key_words'])
            except:
                doc['key_words'] = [k.strip() for k in doc['key_words'].split(',')]
        
        # Repair metadata
        if isinstance(doc.get('metadata'), str):
            try:
                doc['metadata'] = ast.literal_eval(doc['metadata'])
            except:
                doc['metadata'] = {}

        # Load cleaned text into memory if it exists on disk
        # User requirement: Avoid repeated disk reads by having it in RAM
        c_path = doc.get('cleaned_text_path')
        if c_path and os.path.exists(c_path):
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    doc['cleaned_text'] = f.read()
            except Exception as e:
                print(f"Warning: Could not load cleaned text for {doc.get('doc_id')}: {e}")
                
        docs.append(doc)
    
    return docs

def build_resume_graph():
    """Builds a graph starting from Agent 2."""
    graph = StateGraph(dict)
    graph.add_node("agent2", run_agent2)
    graph.add_node("agent3", run_agent3)
    
    graph.set_entry_point("agent2")
    graph.add_edge("agent2", "agent3")
    
    return graph.compile()

def run_resume_pipeline():
    try:
        raw_docs = load_previous_results()
        print(f"Successfully loaded {len(raw_docs)} documents.")
        
        graph = build_resume_graph()
        
        initial_state = {
            "raw_docs": raw_docs
        }
        
        print("Starting pipeline from Agent 2...")
        final_state = graph.invoke(initial_state)
        print("Pipeline execution finished successfully.")
        
    except Exception as e:
        print(f"Error resuming pipeline: {e}")

if __name__ == "__main__":
    run_resume_pipeline()
