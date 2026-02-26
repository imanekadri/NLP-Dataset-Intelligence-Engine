from graph.workflow import build_graph
from core.logger import setup_logger

def main():
    setup_logger()

    graph = build_graph()

    initial_state = {
        "metadata": {
            "num_documents": 10000,
            "languages": ["en"],
            "avg_length": 450,
            "structure_type": "news_article",
            "topics_detected": ["politics", "world"],
            "label_column_detected": True,
            "unique_labels": 2,
            "label_names": ["True", "Fake"],
            "pii_detected": False,
            "source_format": "csv"
        },
        "sample_texts": [
            "Breaking news: government election results",
            "The president announced new policies today",
            "Political debates continue worldwide"
        ],
        "enriched_metadata": {},
        "brain_decision": {},
        "errors": []
    }


    result = graph.invoke(initial_state)

    print("=== FINAL OUTPUT ===")
    print(result["brain_decision"])

if __name__ == "__main__":
    main()


