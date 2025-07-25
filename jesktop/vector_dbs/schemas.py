import base64
from typing import Annotated, Literal

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


class Image(BaseModel):
    """Represents an image stored in the database.

    Attributes:
        id: The ID of the image.
        note_id: The ID of the note the image belongs to.
        content: The image content, encoded as base64.
        mime_type: The MIME type of the image.
        relative_path: The path of the image relative to the note.
        absolute_path: The absolute path of the image.
    """

    id: str
    note_id: str
    content: Annotated[
        bytes,
        BeforeValidator(lambda x: base64.b64decode(x) if isinstance(x, str) else x),
        PlainSerializer(lambda x: base64.b64encode(x).decode(), return_type=str),
    ]
    mime_type: str
    relative_path: str
    absolute_path: str


class NoteRelationship(BaseModel):
    """Represents a relationship between two notes."""

    source_note_id: str
    target_note_id: str
    relationship_type: Literal["wikilink", "embed", "folder_sibling", "temporal"]
    context: str = ""  # surrounding text where link appears
    strength: float = 1.0  # relationship weight/frequency


class RelationshipGraph(BaseModel):
    """Represents the complete relationship graph for all notes."""

    relationships: list[NoteRelationship] = []
    note_clusters: dict[str, list[str]] = {}  # topic/folder-based groupings
