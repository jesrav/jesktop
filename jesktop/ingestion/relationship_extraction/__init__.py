"""Relationship extraction module for processing note links and building relationship graphs."""

from jesktop.ingestion.relationship_extraction.graph_builder import RelationshipGraphBuilder
from jesktop.ingestion.relationship_extraction.resolver import ReferenceResolver

__all__ = [
    "ReferenceResolver",
    "RelationshipGraphBuilder",
]
