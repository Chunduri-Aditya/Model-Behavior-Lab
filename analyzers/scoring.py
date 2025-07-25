def score_logic(output):
    if "yes" in output.lower() or "no" in output.lower():
        return 1
    return 0

def score_emotion(output):
    if any(word in output.lower() for word in ["feel", "sunrise", "hope", "beauty", "peace"]):
        return 1
    return 0

def score_hallucination(output):
    if "mars" in output.lower() and any(name in output.lower() for name in ["elon", "martian", "john", "xyz"]):
        return 0
    if "nobody" in output.lower() or "not applicable" in output.lower():
        return 1
    return 0.5

def score_code(output):
    return 1 if "def" in output and "return" in output else 0