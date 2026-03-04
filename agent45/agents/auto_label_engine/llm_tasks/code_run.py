def run(llm, text):
    prompt = f"Generate code examples and explanation for:\n{text}"
    return llm.generate(prompt)