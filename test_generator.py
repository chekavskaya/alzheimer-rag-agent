import os
from rag.generator import Generator


API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Please set it as an environment variable."
    )


context = (
    "Alzheimer's disease involves amyloid-beta aggregation and tau pathology, "
    "which are considered key therapeutic targets [1]."
)

gen = Generator(api_key=API_KEY)

answer = gen.generate(
    question="What are potential therapeutic targets for Alzheimer's disease?",
    context=context,
)

print("ANSWER:\n", answer)
