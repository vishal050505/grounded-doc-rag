from groq import Groq


class AnswerGenerator:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate(self, question, retrieved_docs):
        # Build context
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_blocks.append(
                f"[Source {i}]\n{doc['text']}"
            )
        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a STRICT document-grounded answer extractor.

CRITICAL RULES (NO EXCEPTIONS):
- Use ONLY the text provided in the sources.
- DO NOT add headings, section numbers, or lecture titles.
- DO NOT summarize chapters or explain broadly.
- DO NOT rephrase unless absolutely necessary.
- DO NOT introduce new terms.
- If the exact answer is NOT explicitly stated, reply with:
  "The document does not explicitly answer this question."

OUTPUT RULES:
- Answer in 1–3 sentences maximum.
- Answer must be factual and directly supported by the text.
- Do NOT mention section numbers like 1.1.1, Lecture 1, etc.

SOURCES:
{context}

QUESTION:
{question}

FINAL ANSWER (nothing else):
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

        # Evidence = top chunks (already validated)
        evidence = [doc["text"][:200] for doc in retrieved_docs[:2]]
        citations = list({doc["source"] for doc in retrieved_docs})

        return answer, evidence, citations
