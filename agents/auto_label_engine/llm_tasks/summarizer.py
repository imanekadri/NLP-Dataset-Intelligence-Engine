def run(llm, text):
    prompt = f"Summarize this text:\n{text}"
    return llm.generate(prompt)