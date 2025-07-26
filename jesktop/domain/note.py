"""Note domain models."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, PlainSerializer


class Note(BaseModel):
    """Represents a full, non-chunked note."""

    id: str
    title: str
    path: str
    content: str
    metadata: dict
    outbound_links: list[str] = []  # note_ids this note links to
    inbound_links: list[str] = []  # note_ids that link to this note
    embedded_content: list[str] = []  # image/drawing hashes referenced
    tags: list[str] = []  # extracted from content/path
    folder_path: str = ""  # for hierarchical relationships


class Chunk(BaseModel):
    """Represents a chunk of text from a note for vector search."""

    id: int
    note_id: str
    title: str
    text: str
    start_pos: int
    end_pos: int


def nd_array_before_validator(x: list[float]) -> NDArray[np.float32]:
    return np.array(x, dtype=np.float32)


def nd_array_serializer(x: NDArray[np.float32]) -> list[float]:
    return x.tolist()  # type: ignore


NumPyArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_before_validator),
    PlainSerializer(nd_array_serializer, return_type=list),
]


class EmbeddedChunk(Chunk):
    vector: NumPyArray

    class Config:
        arbitrary_types_allowed = True
