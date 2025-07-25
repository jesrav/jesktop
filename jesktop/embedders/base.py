from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
