"""Reference resolution for converting note names/paths to note IDs."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReferenceResolver:
    """Handles resolution of note references from wikilinks to note IDs."""

    def __init__(self, note_mapping: dict[str, str]):
        """Initialize resolver with a note mapping.

        Args:
            note_mapping: Dictionary mapping note names/stems to note IDs/asset references
        """
        self.note_mapping = note_mapping

    def resolve_references(self, links: list[str]) -> list[str]:
        """Convert note names/paths to note IDs using the mapping dictionary.

        Args:
            links: List of note names or paths from wikilinks

        Returns:
            List of resolved note IDs and asset references
        """
        resolved_ids = []
        for link in links:
            resolved_id = self._resolve_single_reference(link)
            if resolved_id:
                resolved_ids.append(resolved_id)
        return resolved_ids

    def _resolve_single_reference(self, link: str) -> str | None:
        """Resolve a single reference to a note ID or asset reference.

        Args:
            link: Note name or path from wikilink

        Returns:
            Resolved note ID/asset reference or None if not found
        """
        # Try exact match first
        if link in self.note_mapping:
            return self.note_mapping[link]

        # Try with .md extension
        md_link = f"{link}.md"
        if md_link in self.note_mapping:
            return self.note_mapping[md_link]

        # Try as filename stem
        for path, note_id in self.note_mapping.items():
            if Path(path).stem == link:
                return note_id

        # Check if it might be an image or excalidraw file with different casing
        # or in a different location - be more lenient for assets
        link_lower = link.lower()
        for path, asset_id in self.note_mapping.items():
            if asset_id.startswith(("image:", "excalidraw:")) and (
                path.lower() == link_lower or Path(path).stem.lower() == link_lower
            ):
                return asset_id

        logger.warning(f"Could not resolve wikilink: {link}")
        return None
