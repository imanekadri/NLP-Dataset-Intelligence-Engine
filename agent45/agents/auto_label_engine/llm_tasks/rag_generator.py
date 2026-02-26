def run(llm, question, context):
    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer clearly:
    """
    return llm.generate(prompt)