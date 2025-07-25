from pathlib import Path
from typing import Dict, List

import numpy as np

from jesktop.vector_dbs.base import VectorDB
from jesktop.vector_dbs.schemas import Chunk, Note


class FakeVectorDB(VectorDB):
    """Fake vector DB with predefined notes and chunks."""

    def __init__(self, notes: Dict[str, Note]) -> None:
        self._notes = notes

    def get_closest_chunks(self, input_vector: np.ndarray, closest: int) -> List[Chunk]:
        return [
            Chunk(
                id=1,
                note_id="note1",
                title="Test Note 1",
                text="Test chunk 1",
                start_pos=0,
                end_pos=10,
            ),
            Chunk(
                id=2,
                note_id="note2",
                title="Test Note 2",
                text="Test chunk 2",
                start_pos=0,
                end_pos=10,
            ),
        ]

    def get_note(self, note_id: str) -> Note | None:
        try:
            return self._notes[note_id]
        except KeyError:
            return None

    def get_related_notes(self, note_id: str, max_depth: int = 2) -> List[Note]:
        return []

    def get_note_cluster(self, note_id: str) -> List[Note]:
        return []

    def find_path_between_notes(self, source_id: str, target_id: str) -> List[str]:
        return []

    def get_relationship_context(self, source_id: str, target_id: str) -> str:
        return ""

    def find_note_by_title(self, title: str) -> Note | None:
        """Find note by title for testing."""
        # Simple implementation for testing
        for note in self._notes.values():
            if note.title and note.title.lower() == title.lower():
                return note
            note_stem = Path(note.path).stem
            if note_stem.lower() == title.lower():
                return note
        return None

    def save(self, filepath: str) -> None:
        """Save the vector database to disk."""
        pass

    @classmethod
    def load(cls, filepath: str) -> "FakeVectorDB":
        """Load a vector database from disk."""
        return cls({})
