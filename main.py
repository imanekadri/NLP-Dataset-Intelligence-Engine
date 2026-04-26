from agents.export_engine.export_agent import ExportEngineAgent

state = {
    "structured_dataset": [
        {
            "text": "cancel my order",
            "intent": "cancel_order",
            "entities": ["order"],
            "sentiment": "negative"
        }
    ],
    "dataset_type": "chatbot_training",
    "ml_format": "instruction_tuning"
}

agent = ExportEngineAgent()

state = agent.run(state)

print(state["exports"]["formats"])