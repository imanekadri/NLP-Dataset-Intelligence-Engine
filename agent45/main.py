# from graph.workflow import build_graph
# from core.logger import setup_logger
#
# def main():
#     setup_logger()
#
#     graph = build_graph()
#
#     initial_state = {
#         "metadata": {
#             "num_documents": 10000,
#             "languages": ["en"],
#             "avg_length": 450,
#             "structure_type": "news_article",
#             "topics_detected": ["politics", "world"],
#             "label_column_detected": True,
#             "unique_labels": 2,
#             "label_names": ["True", "Fake"],
#             "pii_detected": False,
#             "source_format": "csv"
#         },
#         "sample_texts": [
#             "Breaking news: government election results",
#             "The president announced new policies today",
#             "Political debates continue worldwide"
#         ],
#         "enriched_metadata": {},
#         "brain_decision": {},
#         "errors": []
#     }
#
#
#     result = graph.invoke(initial_state)
#
#     print("=== FINAL OUTPUT ===")
#     print(result["brain_decision"])
#
# if __name__ == "__main__":
#     main()
#
#




from agent45.agents.auto_label_engine.agent import AutoLabelEngine


def main():

    engine = AutoLabelEngine()

    brain_decision = {
        "dataset_type": "chatbot_training",
        "tasks": ["intent", "ner", "sentiment"],
        "labels": ["cancel_order", "refund", "complaint"]
    }

    text = "je veux annuler ma commande"

    result = engine.process(text, brain_decision)

    print(result)


if __name__ == "__main__":
    main()

#
# from agents.auto_label_engine.agent import AutoLabelEngine
# import json
#
#
# def pretty_print(title, data):
#     print("\n" + "=" * 60)
#     print(f"🔎 {title}")
#     print("=" * 60)
#     print(json.dumps(data, indent=4, ensure_ascii=False))
#
#
# def test_individual_tasks(engine, text):
#     """
#     Tests classical NLP tasks independently
#     """
#
#     tasks = [
#         "classification",
#         "ner",
#         "intent",
#         "sentiment",
#
#     ]
#
#     brain_decision = {
#         "tasks": tasks,
#         "labels": [
#
#             "This message is a greeting",
#             "This message is a request to cancel an order",
#             "This message is a customer complaint",
#             "This message is a refund request"
#
#         ]
#     }
#
#     result = engine.process(text, brain_decision)
#
#     pretty_print("INDIVIDUAL TASKS RESULT", result)
#
#
# def test_translation(engine, text):
#     brain_decision = {
#         "tasks": [],
#     }
#
#     result = engine.process(
#         text,
#         brain_decision,
#         dataset_type="translation"
#     )
#
#     pretty_print("TRANSLATION RESULT", result)
#
#
# def test_llm_tasks(engine, text):
#     llm_tests = [
#         "resume",
#         "chatbot_training",
#         "qa",
#         "instruction_tuning",
#         "code_dataset",
#         "rag"
#     ]
#
#     for dataset_type in llm_tests:
#         print("\n" + "#" * 60)
#         print(f"🚀 Testing LLM Task: {dataset_type}")
#         print("#" * 60)
#
#         brain_decision = {
#             "tasks": [],
#             "context": "La commande numéro 123456 a été envoyée hier."  # For RAG
#         }
#
#         result = engine.process(
#             text,
#             brain_decision,
#             dataset_type=dataset_type
#         )
#
#         pretty_print(f"{dataset_type.upper()} RESULT", result)
#
#
# def test_edge_cases(engine):
#     edge_inputs = [
#         "",
#         "1234567890",
#         "Merci beaucoup !!!",
#         "I want to cancel my order",
#         "Annuler",
#     ]
#
#     for text in edge_inputs:
#         print("\n" + "-" * 60)
#         print(f"⚠ Edge Case Input: '{text}'")
#         print("-" * 60)
#
#         brain_decision = {
#             "tasks": ["intent", "sentiment"]
#         }
#
#         result = engine.process(text, brain_decision)
#         pretty_print("EDGE RESULT", result)
#
#
# def main():
#     print("\n🚀 Initializing AutoLabelEngine...")
#     engine = AutoLabelEngine()
#
#     test_text = "Je veux annuler ma commande numéro 123456"
#
#     # =====================================================
#     # 🔹 Classical NLP Tasks
#     # =====================================================
#     test_individual_tasks(engine, test_text)
#
#     # =====================================================
#     # 🔹 Translation
#     # =====================================================
#     test_translation(engine, test_text)
#
#     # =====================================================
#     # 🔹 Generative LLM Tasks
#     # =====================================================
#     test_llm_tasks(engine, test_text)
#
#     # =====================================================
#     # 🔹 Edge Case Testing
#     # =====================================================
#     test_edge_cases(engine)
#
#     print("\n✅ ALL TESTS COMPLETED")
#
#
# if __name__ == "__main__":
#     main()