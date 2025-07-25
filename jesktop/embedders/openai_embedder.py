import numpy as np
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str):
        self.openai_client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        embedding = (
            self.openai_client.embeddings.create(input=text, model="text-embedding-3-large")
            .data[0]
            .embedding
        )
        return np.array(embedding, dtype=np.float32)
