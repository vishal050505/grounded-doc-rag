def compute_confidence(retrieved_chunks):
    if not retrieved_chunks:
        return 0.0

    scores = [chunk["score"] for chunk in retrieved_chunks]

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    return {
        "average": round(avg_score, 2),
        "max": round(max_score, 2)
    }
