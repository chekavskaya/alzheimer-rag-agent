import requests


class Generator:
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(
        self,
        question: str,
        context: str,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        
        prompt = f"""
You are a biomedical research assistant.

Use ONLY the following scientific context to answer the question.
Cite sources using [number] notation when relevant.
Do not add information that is not supported by the context.

Context:
{context}

Question:
{question}

Answer:
""".strip()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()
