import numpy as np

from jesktop.embedders.base import Embedder


class FakeEmbedder(Embedder):
    """Fake embedder that returns zero vectors."""

    def embed(self, text: str) -> np.ndarray:
        return np.zeros(10)
