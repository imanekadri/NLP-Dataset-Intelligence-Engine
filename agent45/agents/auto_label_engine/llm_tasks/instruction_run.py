def run(llm, text):
    prompt = f"""
    Convert this into instruction-tuning dataset format:
    {text}
    """
    return llm.generate(prompt)