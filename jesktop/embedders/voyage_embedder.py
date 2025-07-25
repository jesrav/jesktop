import numpy as np
import voyageai


class VoyageEmbedder:
    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        result = self.client.embed(texts=[text], model="voyage-3", input_type="document")
        embedding = result.embeddings[0]
        return np.array(embedding, dtype=np.float32)
