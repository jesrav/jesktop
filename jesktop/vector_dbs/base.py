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

    def add_chunk(self, chunk: EmbeddedChunk) -> None:
        """Add an embedded chunk to the database."""
        ...

    def update_relationship_graph(self, relationship_graph: RelationshipGraph) -> None:
        """Update the relationship graph."""
        ...

    def update_note(self, note: Note) -> None:
        """Add a new note or update an existing one."""
        ...

    def delete_note(self, note_id: str) -> None:
        """Delete a note and all its associated chunks."""
        ...

    def delete_chunks_for_note(self, note_id: str) -> None:
        """Delete all chunks associated with a note."""
        ...

    def get_all_note_ids(self) -> set[str]:
        """Get all note IDs in the database."""
        ...

    def get_notes_by_ids(self, note_ids: list[str]) -> dict[str, Note]:
        """Get multiple notes by their IDs, returning a dictionary mapping ID to Note.

        Args:
            note_ids: List of note IDs to retrieve

        Returns:
            Dictionary mapping note_id to Note for all found notes
        """
        ...

    def clear(self) -> None:
        """Clear all data from the database."""
        ...
