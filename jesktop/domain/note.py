"""Note domain models."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, PlainSerializer


class Note(BaseModel):
    """Represents a full, non-chunked note.

    Attributes:
        id: Unique identifier (MD5 hash of relative file path)
        title: Note title extracted from filename or first header
        path: Absolute file path
        content: Full markdown content
        created: File creation timestamp (seconds since epoch)
        modified: File modification timestamp (seconds since epoch)
        outbound_links: List of note IDs this note links to
        inbound_links: List of note IDs that link to this note
        embedded_content: List of image/drawing hashes referenced
        tags: List of tags extracted from content/path
        folder_path: Relative folder path for hierarchical relationships
    """

    id: str
    title: str
    path: str
    content: str
    created: float
    modified: float
    outbound_links: list[str] = []
    inbound_links: list[str] = []
    embedded_content: list[str] = []
    tags: list[str] = []
    folder_path: str = ""


class Chunk(BaseModel):
    """Represents a chunk of text from a note for vector search."""

    id: str  # Changed to string for stable IDs like "note_id_0"
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

    model_config = {"arbitrary_types_allowed": True}
