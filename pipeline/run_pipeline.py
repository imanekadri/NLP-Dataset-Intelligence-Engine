from pipeline.graph import build_graph

def run_pipeline():
    graph = build_graph()

    initial_state = {}
    final_state = graph.invoke(initial_state)

    print("The pipeline is done")
    

if __name__ == "__main__":
    run_pipeline()