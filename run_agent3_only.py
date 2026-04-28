import os
import json
import pandas as pd
from agents.agent3_und import run_agent3

def run_only_agent3():
    # Load raw_docs from agent1's trace_index.csv if it exists
    trace_path = "output/agent1/trace_index.csv"
    if not os.path.exists(trace_path):
        print(f"Error: {trace_path} not found. Run Agent 1 first.")
        return

    print(f"Loading data from {trace_path}...")
    df = pd.read_csv(trace_path)
    # Convert dataframe to list of dicts as expected by Agent 3
    raw_docs = df.to_dict('records')
    
    # We also need 'topic' for each doc, which usually comes from Agent 2.
    # If Agent 2 has run, we might find topics in its report.
    agent2_report = "output/agent2/dataset_profile.json"
    if os.path.exists(agent2_report):
        print(f"Enriching with topics from {agent2_report}...")
        with open(agent2_report, 'r') as f:
            profile = json.load(f)
            # Map doc_id to topic if possible
            # Note: Agent 2 structure might vary, let's assume it has topics.
    
    state = {
        "raw_docs": raw_docs
    }
    
    print("Running Agent 3 (Professor's version)...")
    result = run_agent3(state)
    print(f"Agent 3 complete. Results in: {result.get('agent3_results')}")

if __name__ == "__main__":
    run_only_agent3()
