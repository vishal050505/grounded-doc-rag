from groq import Groq


class AnswerGenerator:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate(self, question, retrieved_docs):
        """
        retrieved_docs: list of dicts with keys
        - text
        - source
        - score
        """

        # Build context
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"[Context {i}]\n{doc['text']}\n\n"

       
        # PROMPT 
       
        prompt = f"""
You are a document-based question answering assistant.

Rules:
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- If the answer is not present in the context, clearly say so.

Answering Style:
1. Start with a short Definition if applicable (1–2 lines).
2. If the context includes lists, layers, steps, or components:
   - List their names clearly.
3. Explain only what is explicitly asked.
4. Keep the answer concise, clear, and high quality.
5. Avoid unnecessary verbosity.

Context:
{context}

Question:
{question}

Return the answer in this format:

Answer:
<clear, structured answer>

Evidence:
- <short supporting quote or summary from Context X>
- <short supporting quote or summary from Context Y>
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response.choices[0].message.content.strip()

        # PARSE ANSWER & EVIDENCE
        
        answer = raw_output
        evidence = []

        if "Evidence:" in raw_output:
            answer_part, evidence_part = raw_output.split("Evidence:", 1)
            answer = answer_part.replace("Answer:", "").strip()

            evidence = [
                e.strip("- ").strip()
                for e in evidence_part.strip().split("\n")
                if e.strip()
            ]
        else:
            answer = answer.replace("Answer:", "").strip()

        citations = list({doc["source"] for doc in retrieved_docs})

        return answer, evidence, citations
