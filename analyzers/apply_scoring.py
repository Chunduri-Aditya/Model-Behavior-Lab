from analyzers.scoring import score_logic, score_emotion, score_hallucination, score_code

def apply_scoring(entry):
    task = entry["task"].lower()
    output = entry["output"]

    if "logic" in task:
        return score_logic(output)
    elif "emotion" in task:
        return score_emotion(output)
    elif "hallucination" in task:
        return score_hallucination(output)
    elif "code" in task:
        return score_code(output)
    else:
        return 0  # unknown task, conservative score