import json
from typing import Dict, Optional

from jesktop.domain.image import Image
from jesktop.image_store.base import ImageStore


class LocalImageStore(ImageStore):
    """Local image store that saves images to a JSON file."""

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

    def save_image(self, image: Image) -> None:
        """Save an image to the store."""
        self._images[image.id] = image

    def save(self, filepath: str) -> None:
        """Save the image store to a JSON file."""
        data = {
            "images": {image_id: image.model_dump() for image_id, image in self._images.items()}
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "LocalImageStore":
        """Load an image store from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            images = {
                image_id: Image(**image_data) for image_id, image_data in data["images"].items()
            }
            return cls(images)
