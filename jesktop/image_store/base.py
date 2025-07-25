from typing import Optional, Protocol

from jesktop.vector_dbs.schemas import Image


class ImageStore(Protocol):
    """Protocol for image storage implementations."""

    def get_image(self, image_id: str) -> Image:
        """Get an image by its ID."""
        ...

    def get_image_id_by_path(self, note_id: str, relative_path: str) -> Optional[str]:
        """Get image ID by note ID and relative path."""
        ...

    def save_image(self, image: Image) -> None:
        """Save an image to the store."""
        ...

    def save(self, filepath: str) -> None:
        """Save the image store to disk."""
        ...

    @classmethod
    def load(cls, filepath: str) -> "ImageStore":
        """Load an image store from disk."""
        ...
