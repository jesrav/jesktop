import json
import os
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np

from jesktop.vector_dbs.base import VectorDB
from jesktop.vector_dbs.schemas import Chunk, EmbeddedChunk, Note, RelationshipGraph


class LocalVectorDB(VectorDB):
    """Local vector database that stores embeddings in a JSON file."""

    def __init__(
        self,
        notes: Dict[str, Note] | None = None,
        embedded_chunks: Dict[int, EmbeddedChunk] | None = None,
        relationship_graph: RelationshipGraph | None = None,
    ) -> None:
        self._notes = notes if notes is not None else {}
        self._embedded_chunks = embedded_chunks if embedded_chunks is not None else {}
        self._relationship_graph = (
            relationship_graph if relationship_graph is not None else RelationshipGraph()
        )

    def get_closest_chunks(self, input_vector: np.ndarray, closest: int) -> List[Chunk]:
        """Get the closest chunks to an input vector."""
        # Convert input vector to numpy array
        input_vector = np.array(input_vector)

        # Calculate cosine similarity with all vectors
        similarities = []
        for chunk in self._embedded_chunks.values():
            chunk_vector = np.array(chunk.vector)
            similarity = np.dot(input_vector, chunk_vector) / (
                np.linalg.norm(input_vector) * np.linalg.norm(chunk_vector)
            )
            similarities.append((similarity, chunk))

        # Sort by similarity and return top k chunks
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [
            Chunk(
                id=chunk.id,
                note_id=chunk.note_id,
                title=chunk.title,
                text=chunk.text,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
            )
            for _, chunk in similarities[:closest]
        ]

    def get_note(self, note_id: str) -> Note | None:
        """Get a note by its ID."""
        return self._notes.get(note_id)

    def get_related_notes(self, note_id: str, max_depth: int = 2) -> List[Note]:
        """Get notes related to the given note through relationships."""
        if note_id not in self._notes:
            return []

        visited = set()
        related_notes = []
        queue = deque([(note_id, 0)])  # (note_id, depth)
        visited.add(note_id)

        while queue:
            current_id, depth = queue.popleft()

            if depth > 0:  # Don't include the source note itself
                if current_id in self._notes:
                    related_notes.append(self._notes[current_id])

            if depth < max_depth:
                # Add outbound links
                current_note = self._notes.get(current_id)
                if current_note:
                    for linked_id in current_note.outbound_links + current_note.inbound_links:
                        if linked_id not in visited:
                            visited.add(linked_id)
                            queue.append((linked_id, depth + 1))

        return related_notes

    def get_note_cluster(self, note_id: str) -> List[Note]:
        """Get all notes in the same folder/cluster as the given note."""
        note = self._notes.get(note_id)
        if not note or not note.folder_path:
            return []

        cluster_note_ids = self._relationship_graph.note_clusters.get(note.folder_path, [])
        return [
            self._notes[nid] for nid in cluster_note_ids if nid in self._notes and nid != note_id
        ]

    def find_path_between_notes(self, source_id: str, target_id: str) -> List[str]:
        """Find the shortest path between two notes through relationships."""
        if source_id not in self._notes or target_id not in self._notes:
            return []

        if source_id == target_id:
            return [source_id]

        visited = set()
        queue = deque([(source_id, [source_id])])  # (note_id, path)
        visited.add(source_id)

        while queue:
            current_id, path = queue.popleft()
            current_note = self._notes.get(current_id)

            if current_note:
                # Check all connected notes
                connected_ids = current_note.outbound_links + current_note.inbound_links
                for connected_id in connected_ids:
                    if connected_id == target_id:
                        return path + [connected_id]

                    if connected_id not in visited and connected_id in self._notes:
                        visited.add(connected_id)
                        queue.append((connected_id, path + [connected_id]))

        return []  # No path found

    def get_relationship_context(self, source_id: str, target_id: str) -> str:
        """Get the context text for a relationship between two notes."""
        for rel in self._relationship_graph.relationships:
            if rel.source_note_id == source_id and rel.target_note_id == target_id:
                return rel.context
        return ""

    def find_note_by_title(self, title: str) -> Note | None:
        """Find note by title, supporting fuzzy matching."""
        # Try different matching strategies in order of precision
        for strategy in [
            self._match_exact_title,
            self._match_case_insensitive_title,
            self._match_normalized_title,
            self._match_stem,
            self._match_substring_title,
        ]:
            result = strategy(title)
            if result:
                return result
        return None

    def _match_exact_title(self, title: str) -> Note | None:
        """Match exact title including empty strings."""
        for note in self._notes.values():
            if note.title == title:
                return note
        return None

    def _match_case_insensitive_title(self, title: str) -> Note | None:
        """Match title case insensitively."""
        for note in self._notes.values():
            if note.title and note.title.lower() == title.lower():
                return note
        return None

    def _match_normalized_title(self, title: str) -> Note | None:
        """Match title with space/underscore normalization."""
        normalized_title = title.lower().replace(" ", "_")
        for note in self._notes.values():
            if note.title and note.title.lower().replace(" ", "_") == normalized_title:
                return note
        return None

    def _match_stem(self, title: str) -> Note | None:
        """Match by file stem (filename without extension)."""
        normalized_title = title.lower().replace(" ", "_")
        for note in self._notes.values():
            note_stem = Path(note.path).stem
            if note_stem.lower() in (normalized_title, title.lower()):
                return note
        return None

    def _match_substring_title(self, title: str) -> Note | None:
        """Match by substring in title."""
        for note in self._notes.values():
            if note.title and title.lower() in note.title.lower():
                return note
        return None

    def save(self, filepath: str) -> None:
        """Save the vector database to a JSON file."""
        data = {
            "notes": {note_id: note.model_dump() for note_id, note in self._notes.items()},
            "chunks": {
                chunk_id: chunk.model_dump() for chunk_id, chunk in self._embedded_chunks.items()
            },
            "relationships": self._relationship_graph.model_dump(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "LocalVectorDB":
        """Load a vector database from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        notes = {note_id: Note(**note_data) for note_id, note_data in data["notes"].items()}
        chunks = {
            chunk_id: EmbeddedChunk(**chunk_data) for chunk_id, chunk_data in data["chunks"].items()
        }

        # Load relationships if available (backward compatibility)
        relationship_graph = RelationshipGraph()
        if "relationships" in data:
            relationship_graph = RelationshipGraph(**data["relationships"])

        return cls(notes=notes, embedded_chunks=chunks, relationship_graph=relationship_graph)

    @classmethod
    def load_pickle(cls, path: str) -> "LocalVectorDB":
        """Load the database from disk."""
        db = cls()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                db._notes = data["notes"]
                db._embedded_chunks = {chunk.id: chunk for chunk in data["chunks"]}
        return db
