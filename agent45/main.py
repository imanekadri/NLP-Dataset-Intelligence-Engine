# # from graph.workflow import build_graph
# # from core.logger import setup_logger
# #
# # def main():
# #     setup_logger()
# #
# #     graph = build_graph()
# #
# #     initial_state = {
# #         "metadata": {
# #             "num_documents": 10000,
# #             "languages": ["en"],
# #             "avg_length": 450,
# #             "structure_type": "news_article",
# #             "topics_detected": ["politics", "world"],
# #             "label_column_detected": True,
# #             "unique_labels": 2,
# #             "label_names": ["True", "Fake"],
# #             "pii_detected": False,
# #             "source_format": "csv"
# #         },
# #         "sample_texts": [
# #             "Breaking news: government election results",
# #             "The president announced new policies today",
# #             "Political debates continue worldwide"
# #         ],
# #         "enriched_metadata": {},
# #         "brain_decision": {},
# #         "errors": []
# #     }
# #
# #
# #     result = graph.invoke(initial_state)
# #
# #     print("=== FINAL OUTPUT ===")
# #     print(result["brain_decision"])
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
#
#
#
#
# from agent45.agents.auto_label_engine.agent import AutoLabelEngine
#
# #
# # def main():
# #
# #     engine = AutoLabelEngine()
# #
# #     brain_decision = {
# #         "dataset_type": "chatbot_training",
# #         "tasks": ["intent", "ner", "sentiment"],
# #         "labels": ["cancel_order", "refund", "complaint"]
# #     }
# #
# #     text = "je veux annuler ma commande"
# #
# #     result = engine.process(text, brain_decision)
# #
# #     print(result)
# #
# #
# # if __name__ == "__main__":
# #     main()
#

from agents.auto_label_engine.agent import AutoLabelEngine
import json


def pretty_print(title, data):
    print("\n" + "=" * 60)
    print(f"🔎 {title}")
    print("=" * 60)
    print(json.dumps(data, indent=4, ensure_ascii=False))


def test_individual_tasks(engine, text):
    """
    Tests classical NLP tasks independently
    """

    tasks = [
        "classification",
        "ner",
        "intent",
        "sentiment",

    ]

    brain_decision = {
        "tasks": tasks,
        "labels": [

            "This message is a greeting",
            "This message is a request to cancel an order",
            "This message is a customer complaint",
            "This message is a refund request"

        ]
    }

    result = engine.process(text, brain_decision)

    pretty_print("INDIVIDUAL TASKS RESULT", result)


def test_translation(engine, text):
    brain_decision = {
        "tasks": [],
    }

    result = engine.process(
        text,
        brain_decision,
        dataset_type="translation"
    )

    pretty_print("TRANSLATION RESULT", result)


def test_llm_tasks(engine, text):
    llm_tests = [
        "summarizer",
        "chatbot_run",
        "qa_run",
        "instruction_run",
        "code_run",
        "rag_run"
    ]

    for dataset_type in llm_tests:
        print("\n" + "#" * 60)
        print(f"🚀 Testing LLM Task: {dataset_type}")
        print("#" * 60)

        brain_decision = {
            "tasks": [],
            "context": "La commande numéro 123456 a été envoyée hier."  # For RAG
        }

        result = engine.process(
            text,
            brain_decision,
            dataset_type=dataset_type
        )

        pretty_print(f"{dataset_type.upper()} RESULT", result)


def test_edge_cases(engine):
    edge_inputs = [
        "",
        "1234567890",
        "Merci beaucoup !!!",
        "I want to cancel my order",
        "Annuler",
    ]

    for text in edge_inputs:
        print("\n" + "-" * 60)
        print(f"⚠ Edge Case Input: '{text}'")
        print("-" * 60)

        brain_decision = {
            "tasks": ["intent", "sentiment"]
        }

        result = engine.process(text, brain_decision)
        pretty_print("EDGE RESULT", result)


def main():
    print("\n🚀 Initializing AutoLabelEngine...")
    engine = AutoLabelEngine()

    test_text = "Je veux annuler ma commande numéro 123456"

    # =====================================================
    # 🔹 Classical NLP Tasks
    # =====================================================
    test_individual_tasks(engine, test_text)

    # =====================================================
    # 🔹 Translation
    # =====================================================
    test_translation(engine, test_text)

    # =====================================================
    # 🔹 Generative LLM Tasks
    # =====================================================
    test_llm_tasks(engine, test_text)

    # =====================================================
    # 🔹 Edge Case Testing
    # =====================================================
    test_edge_cases(engine)

    print("\n✅ ALL TESTS COMPLETED")


if __name__ == "__main__":
    main()

#
#
# """
# test_engine.py — Improved test suite for AutoLabelEngine
#
# Shows all fixes:
#   1. Classification: short label names → correct results
#   2. NER: validated entities → no more garbage (veux=ORG, 123456=DATE)
#   3. Confidence: real scores → no more static 0.5
#   4. LLM tasks: structured JSON → not raw strings
#   5. Batch: process_batch() with pre-fitted BERTopic
#   6. Edge cases: empty, numeric, single word
# """
#
# import json
# from agents.auto_label_engine.agent import AutoLabelEngine
#
#
# def pretty(title: str, data: dict):
#     print("\n" + "=" * 60)
#     print(f"🔎  {title}")
#     print("=" * 60)
#     print(json.dumps(data, indent=4, ensure_ascii=False))
#
#
# def test_classification_fix(engine):
#     """
#     Demonstrates the label-cleaning fix.
#     OLD: labels as sentences → wrong results
#     NEW: short label names → correct results
#     """
#     text = "Je veux annuler ma commande numéro 123456"
#
#     print("\n" + "─" * 60)
#     print("🔧 FIX: Classification label format")
#     print("─" * 60)
#
#     # ❌ OLD: full-sentence labels caused wrong classifications
#     old_brain = {
#         "tasks": ["classification"],
#         "labels": [
#             "This message is a greeting",
#             "This message is a request to cancel an order",
#             "This message is a customer complaint",
#             "This message is a refund request"
#         ]
#     }
#     result_old = engine.process(text, old_brain)
#     print(f"\n❌ OLD labels (sentences): got → '{result_old['classification']['label']}'")
#     print(f"   confidence: {result_old['classification']['confidence']}")
#
#     # ✅ NEW: short descriptive labels → model works correctly
#     new_brain = {
#         "tasks": ["classification"],
#         "labels": [
#             "greeting",
#             "order cancellation request",
#             "customer complaint",
#             "refund request"
#         ]
#     }
#     result_new = engine.process(text, new_brain)
#     print(f"\n✅ NEW labels (short):    got → '{result_new['classification']['label']}'")
#     print(f"   confidence: {result_new['classification']['confidence']}")
#     if result_new['classification'].get('all_scores'):
#         print(f"   all_scores: {result_new['classification']['all_scores']}")
#
#
# def test_ner_fix(engine):
#     """
#     Demonstrates the NER validation fix.
#     OLD: "veux"=ORG, "ma commande"=PERSON, "123456"=DATE
#     NEW: only real entities returned
#     """
#     text = "Je veux annuler ma commande numéro 123456"
#
#     print("\n" + "─" * 60)
#     print("🔧 FIX: NER garbage entity filtering")
#     print("─" * 60)
#
#     brain = {"tasks": ["ner"], "language": "fr"}
#     result = engine.process(text, brain)
#
#     entities = result.get("entities", [])
#     print(f"\nEntities found: {len(entities)}")
#     if entities:
#         for e in entities:
#             print(f"  • '{e['text']}' → {e['label']} ({e.get('description', '')})")
#     else:
#         print("  → No real named entities found in this sentence (correct!)")
#         print("  → '123456' is NOT labeled as DATE (fixed)")
#         print("  → 'veux' is NOT labeled as ORG (fixed)")
#
#
# def test_confidence_fix(engine):
#     """
#     Demonstrates real confidence scoring vs original static 0.5.
#     """
#     print("\n" + "─" * 60)
#     print("🔧 FIX: Real confidence scoring")
#     print("─" * 60)
#
#     test_cases = [
#         ("",                          {"tasks": ["intent", "sentiment"]}, "empty string"),
#         ("I want to cancel my order", {"tasks": ["intent", "sentiment", "ner"]}, "clear English"),
#         ("Je veux annuler ma commande numéro 123456",
#          {"tasks": ["classification", "ner", "intent", "sentiment"],
#           "labels": ["greeting", "order cancellation", "complaint", "refund request"]},
#          "full French"),
#         ("Annuler",                   {"tasks": ["intent", "sentiment"]}, "single word"),
#         ("1234567890",                {"tasks": ["intent", "sentiment"]}, "numbers only"),
#     ]
#
#     for text, brain, desc in test_cases:
#         result = engine.process(text, brain)
#         score = result["confidence_score"]
#         review = "⚠️ needs review" if result.get("needs_review") else "✅ ok"
#         print(f"\n  [{desc}]")
#         print(f"  text: '{text[:50]}'")
#         print(f"  confidence: {score} {review}")
#
#
# def test_individual_tasks(engine):
#     """All NLP tasks on a realistic French support message."""
#     text = "Je veux annuler ma commande numéro 123456"
#
#     brain = {
#         "tasks": ["classification", "ner", "intent", "sentiment", "topic"],
#         "labels": [
#             "greeting",
#             "order cancellation request",
#             "customer complaint",
#             "refund request"
#         ],
#         "language": "fr"
#     }
#
#     result = engine.process(text, brain)
#     pretty("INDIVIDUAL TASKS (improved)", result)
#
#
# def test_translation(engine):
#     text = "Je veux annuler ma commande numéro 123456"
#     result = engine.process(text, {"tasks": []}, dataset_type="translation")
#     pretty("TRANSLATION", result)
#
#
# def test_llm_tasks(engine):
#     """LLM tasks now return structured JSON, not raw strings."""
#     text = "Je veux annuler ma commande numéro 123456"
#
#     llm_tests = [
#         ("resume",             {"tasks": [], "context": ""}),
#         ("chatbot_training",   {"tasks": [], "context": ""}),
#         ("qa",                 {"tasks": [], "context": ""}),
#         ("instruction_tuning", {"tasks": [], "context": ""}),
#         ("rag",                {"tasks": [], "context": "La commande 123456 a été envoyée hier."}),
#     ]
#
#     for dtype, brain in llm_tests:
#         print("\n" + "#" * 60)
#         print(f"🚀 LLM Task: {dtype}")
#         print("#" * 60)
#         result = engine.process(text, brain, dataset_type=dtype)
#         pretty(f"{dtype.upper()} RESULT", result)
#
#         # Verify JSON structure (not raw strings)
#         for key in ["summary", "chatbot_response", "qa_pairs", "instruction_format", "rag_answer"]:
#             if key in result:
#                 value = result[key]
#                 is_structured = isinstance(value, (dict, list))
#                 print(f"  → {key}: {'✅ structured JSON' if is_structured else '❌ raw string'}")
#
#
# def test_batch(engine):
#     """Demonstrates process_batch() with pre-fitted BERTopic."""
#     texts = [
#         "Je veux annuler ma commande",
#         "I want a refund for my purchase",
#         "My package hasn't arrived yet",
#         "I can't log in to my account",
#         "The product quality is terrible",
#         "Thank you for the great service!",
#         "How do I reset my password?",
#         "I was charged twice for the same item",
#         "Where is my delivery?",
#         "I'd like to cancel my subscription",
#     ]
#
#     brain = {
#         "tasks": ["intent", "sentiment", "topic", "clustering"],
#         "language": "fr"
#     }
#
#     print("\n" + "=" * 60)
#     print("🔎  BATCH PROCESSING (10 texts)")
#     print("=" * 60)
#
#     results = engine.process_batch(texts, brain)
#
#     for r in results:
#         intent    = r.get("intent", {}).get("intent", "?")
#         sentiment = r.get("sentiment", {}).get("label", "?")
#         conf      = r.get("confidence_score", 0)
#         review    = "⚠️" if r.get("needs_review") else "✅"
#         text_short = r["text"][:45]
#         print(f"  {review} '{text_short}' | intent={intent} | sentiment={sentiment} | conf={conf:.2f}")
#
#     print(f"\n  LLM stats: {engine.get_stats()}")
#
#
# def test_edge_cases(engine):
#     """Edge cases with correct expected behavior."""
#     cases = [
#         ("",                   "empty string"),
#         ("1234567890",         "numbers only"),
#         ("Merci beaucoup !!!", "short positive fr"),
#         ("I want to cancel my order", "clear English intent"),
#         ("Annuler",            "single French word"),
#     ]
#
#     brain = {"tasks": ["intent", "sentiment"]}
#
#     print("\n" + "=" * 60)
#     print("🔎  EDGE CASES")
#     print("=" * 60)
#
#     for text, desc in cases:
#         result = engine.process(text, brain)
#         intent    = result.get("intent", {}).get("intent", "?")
#         sentiment = result.get("sentiment", {}).get("label", "?")
#         conf      = result.get("confidence_score", 0)
#         review    = "⚠️ needs_review" if result.get("needs_review") else "✅"
#
#         print(f"\n  [{desc}] '{text}'")
#         print(f"    intent={intent} | sentiment={sentiment} | confidence={conf:.4f} {review}")
#
#
# def main():
#     print("\n🚀 Initializing AutoLabelEngine...")
#     engine = AutoLabelEngine()
#
#     # 1. Show classification fix
#     test_classification_fix(engine)
#
#     # 2. Show NER fix
#     test_ner_fix(engine)
#
#     # 3. Show confidence scoring fix
#     test_confidence_fix(engine)
#
#     # 4. Full individual tasks
#     test_individual_tasks(engine)
#
#     # 5. Translation
#     test_translation(engine)
#
#     # 6. LLM tasks (structured JSON)
#     test_llm_tasks(engine)
#
#     # 7. Batch processing
#     test_batch(engine)
#
#     # 8. Edge cases
#     test_edge_cases(engine)
#
#     print("\n✅ ALL TESTS COMPLETED")
#
#
# if __name__ == "__main__":
#     main()