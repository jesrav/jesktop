from typing import Dict, List, Optional

from jesktop.domain.image import Image
from jesktop.image_store.base import ImageStore


class FakeImageStore(ImageStore):
    """Fake image store for testing."""

    def __init__(self, images: Dict[str, Image] | None = None) -> None:
        self._images = images or {}

    def get_image(self, image_id: str) -> Image:
        """Get an image by its ID."""
        if image_id not in self._images:
            raise KeyError(f"Image {image_id} not found")
        return self._images[image_id]

    def get_image_id_by_path(self, note_id: str, relative_path: str) -> Optional[str]:
        """Get image ID by note ID and relative path."""
        for image in self._images.values():
            if image.note_id == note_id and image.relative_path == relative_path:
                return image.id
        return None

    def get_image_ids(self) -> List[str]:
        """Get all image IDs stored in the image store."""
        return list(self._images.keys())

    def add_image(self, image: Image) -> None:
        """Add an image to the store and return its ID."""
        self._images[image.id] = image

    def save(self, filepath: str | None = None) -> None:
        """Save the image store to disk."""
        pass
