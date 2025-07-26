from typing import List, Protocol

import numpy as np

from jesktop.domain.note import Chunk, EmbeddedChunk, Note
from jesktop.domain.relationships import RelationshipGraph


class VectorDB(Protocol):
    def get_closest_chunks(self, input_vector: np.ndarray, closest: int) -> List[Chunk]:
        """Get the closest chunks to an input vector."""
        ...

    def get_note(self, note_id: str) -> Note | None:
        """Get a note by its ID."""
        ...

    def get_related_notes(self, note_id: str, max_depth: int = 2) -> List[Note]:
        """Get notes related to the given note through relationships."""
        ...

    def get_note_cluster(self, note_id: str) -> List[Note]:
        """Get all notes in the same folder/cluster as the given note."""
        ...

    def find_path_between_notes(self, source_id: str, target_id: str) -> List[str]:
        """Find the shortest path between two notes through relationships."""
        ...

    def get_relationship_context(self, source_id: str, target_id: str) -> str:
        """Get the context text for a relationship between two notes."""
        ...

    def find_note_by_title(self, title: str) -> Note | None:
        """Find note by title, supporting fuzzy matching."""
        ...

    def save(self, filepath: str | None = None) -> None:
        """Save the vector database to disk."""
        ...

    def add_note(self, note: Note) -> None:
        """Add a note to the database."""
        ...

    def add_chunk(self, chunk: EmbeddedChunk) -> None:
        """Add an embedded chunk to the database."""
        ...

    def update_relationship_graph(self, relationship_graph: RelationshipGraph) -> None:
        """Update the relationship graph."""
        ...

    def clear(self) -> None:
        """Clear all data from the database."""
        ...
