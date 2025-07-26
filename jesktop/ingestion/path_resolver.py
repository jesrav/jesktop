"""Path resolution for images and attachments."""

from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from loguru import logger


class PathResolver:
    """Resolve image and attachment paths with clear precedence rules."""

    def __init__(self, base_path: Path, attachment_folders: list[str]):
        """
        Initialize PathResolver.

        Args:
            base_path: Base directory for notes (e.g., data/notes)
            attachment_folders: List of attachment folder names (e.g., ["Z - Attachements"])
        """
        self.base_path = Path(base_path)
        self.attachment_folders = attachment_folders

    def resolve_image_path(self, note_file: Path, image_path: str) -> Optional[Path]:
        """
        Resolve image path with clear precedence rules.

        Priority:
        1. Relative to note file (most specific)
        2. In note's asset folder (Note Name.assets/)
        3. In configured attachment folders (Z - Attachements, etc.)
        4. Absolute path in base directory

        Args:
            note_file: Path to the note file referencing the image
            image_path: Image path as found in the note content

        Returns:
            Resolved absolute path if found, None otherwise
        """
        logger.debug(f"Attempting to resolve image path: {image_path}")

        # URL decode first - this fixes the main brittleness issue
        clean_path = unquote(image_path)
        logger.debug(f"URL decoded path: {clean_path}")

        # Try each resolution strategy in order
        resolution_strategies = [
            ("relative to note", self._resolve_relative_to_note),
            ("in note assets folder", self._resolve_in_note_assets),
            ("in attachment folders", self._resolve_in_attachments),
            ("absolute in base", self._resolve_absolute),
        ]

        for strategy_name, resolver_func in resolution_strategies:
            candidate = resolver_func(note_file, clean_path)
            logger.debug(f"Trying {strategy_name}: {candidate}")

            if candidate and candidate.exists():
                logger.info(f"Resolved successfully: {image_path} -> {candidate}")
                return candidate

        logger.warning(f"Failed to resolve image path: {image_path}")
        return None

    def _resolve_relative_to_note(self, note_file: Path, image_path: str) -> Path:
        """Try relative to note file."""
        return note_file.parent / image_path

    def _resolve_in_note_assets(self, note_file: Path, image_path: str) -> Path:
        """Try in note's .assets folder."""
        assets_folder = note_file.parent / f"{note_file.stem}.assets"
        return assets_folder / Path(image_path).name

    def _resolve_in_attachments(self, note_file: Path, image_path: str) -> Optional[Path]:
        """Try in configured attachment folders."""
        for folder in self.attachment_folders:
            # Try direct path in attachment folder
            candidate = self.base_path / folder / image_path
            if candidate.exists():
                return candidate

            # Also try in note-specific asset folder within attachments
            # e.g., "Z - Attachements/Note Name.assets/image.png"
            note_assets_in_attachments = (
                self.base_path / folder / f"{note_file.stem}.assets" / Path(image_path).name
            )
            if note_assets_in_attachments.exists():
                return note_assets_in_attachments

        # Return None if not found in any attachment folder
        return None

    def _resolve_absolute(self, note_file: Path, image_path: str) -> Path:
        """Try absolute path in base directory."""
        return self.base_path / image_path

    def get_resolution_candidates(self, note_file: Path, image_path: str) -> list[Path]:
        """
        Get all candidate paths for debugging purposes.

        Returns:
            List of all paths that would be tried during resolution
        """
        clean_path = unquote(image_path)

        candidates = [
            self._resolve_relative_to_note(note_file, clean_path),
            self._resolve_in_note_assets(note_file, clean_path),
            self._resolve_absolute(note_file, clean_path),
        ]

        # Add attachment folder candidates
        for folder in self.attachment_folders:
            candidates.append(self.base_path / folder / clean_path)

        return candidates
