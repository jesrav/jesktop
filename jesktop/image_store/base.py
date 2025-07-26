from typing import List, Optional, Protocol

from jesktop.domain.image import Image


class ImageStore(Protocol):
    """Protocol for image storage implementations."""

    def get_image(self, image_id: str) -> Image:
        """Get an image by its ID."""
        ...

    def get_image_id_by_path(self, note_id: str, relative_path: str) -> Optional[str]:
        """Get image ID by note ID and relative path."""
        ...

    def get_image_ids(self) -> List[str]:
        """Get all image IDs stored in the image store."""
        ...

    def add_image(self, image: Image) -> None:
        """Add an image to the store."""
        ...

    def save(self, filepath: str | None = None) -> None:
        """Save the image store to disk."""
        ...
