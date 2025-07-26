"""Building relationship graphs from processed notes."""

from jesktop.domain.note import Note
from jesktop.domain.relationships import NoteRelationship, RelationshipGraph

from . import analyzer


class RelationshipGraphBuilder:
    """Builds relationship graphs from processed notes."""

    def build_relationships(self, notes: dict[str, Note]) -> RelationshipGraph:
        """Build relationship graph from processed notes.

        Args:
            notes: Dictionary of note ID to Note objects

        Returns:
            RelationshipGraph with relationships and note clusters
        """
        relationships = self._build_note_relationships(notes)
        note_clusters = self._build_note_clusters(notes)

        return RelationshipGraph(
            relationships=relationships,
            note_clusters=note_clusters,
        )

    def update_inbound_links(
        self, notes: dict[str, Note], relationships: list[NoteRelationship]
    ) -> None:
        """Update inbound_links for all notes based on relationships.

        Args:
            notes: Dictionary of note ID to Note objects
            relationships: List of relationships to process
        """
        # Clear existing inbound links
        for note in notes.values():
            note.inbound_links = []

        # Build inbound links from relationships
        for rel in relationships:
            if rel.target_note_id in notes:
                notes[rel.target_note_id].inbound_links.append(rel.source_note_id)

    def _build_note_relationships(self, notes: dict[str, Note]) -> list[NoteRelationship]:
        """Build relationships from outbound links in notes.

        Args:
            notes: Dictionary of note ID to Note objects

        Returns:
            List of NoteRelationship objects
        """
        relationships = []

        for note in notes.values():
            for target_id in note.outbound_links:
                # Only create note-to-note relationships, skip asset references
                if target_id in notes and not target_id.startswith(("image:", "excalidraw:")):
                    target_note = notes[target_id]

                    # Calculate relationship strength and context
                    strength = analyzer.calculate_relationship_strength(
                        note.content, target_note.title
                    )
                    context = analyzer.extract_relationship_context(note.content, target_note.title)

                    relationship = NoteRelationship(
                        source_note_id=note.id,
                        target_note_id=target_id,
                        relationship_type="wikilink",
                        context=context,
                        strength=strength,
                    )
                    relationships.append(relationship)

        return relationships

    def _build_note_clusters(self, notes: dict[str, Note]) -> dict[str, list[str]]:
        """Build note clusters by folder.

        Args:
            notes: Dictionary of note ID to Note objects

        Returns:
            Dictionary mapping folder paths to lists of note IDs
        """
        note_clusters = {}

        for note in notes.values():
            if note.folder_path:
                if note.folder_path not in note_clusters:
                    note_clusters[note.folder_path] = []
                note_clusters[note.folder_path].append(note.id)

        return note_clusters
