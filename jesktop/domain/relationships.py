"""Relationship domain models."""

from typing import Literal

from pydantic import BaseModel


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
