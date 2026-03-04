def run(llm, text):
    prompt = f"Generate a helpful chatbot response:\nUser: {text}"
    return llm.generate(prompt)