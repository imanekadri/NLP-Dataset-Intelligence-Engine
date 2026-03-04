def run(llm, text):
    prompt = f"Generate question-answer pairs from:\n{text}"
    return llm.generate(prompt)