import requests

class Generator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, question: str, context: str) -> str:
        prompt = f"""You are a specialized Alzheimer's Disease Research Assistant.
Your goal is to identify therapeutic targets and drug modalities (small molecules, biologics, etc.) based ONLY on the provided context.

Context:
{context}

Instruction:
1. Synthesize an answer based on the provided scientific snippets.
2. If the context mentions specific targets (e.g., TREM2, Beta-amyloid, Tau), emphasize them.
3. Categorize the findings if possible (e.g., "Small molecule targets", "Immunotherapy").
4. Use [number] to cite the specific source from the context.
5. If the answer is not in the context, state that clearly.

Question: {question}
Answer:"""

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, 
            "max_tokens": 800
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
