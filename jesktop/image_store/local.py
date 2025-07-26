import json
from pathlib import Path
from typing import List, Optional

from jesktop.domain.image import Image
from jesktop.image_store.base import ImageStore


class LocalImageStore(ImageStore):
    """Local image store that saves images to a JSON file."""

    def __init__(self, filepath: str | Path | None = None) -> None:
        """Initialize LocalImageStore.

        Args:
            filepath: Path to image store file. If provided and exists, will auto-load.
                     If provided and doesn't exist, will save to this path when save() is called.
                     If not provided, creates empty store in memory only.
        """
        self._filepath = str(filepath) if filepath else None

        # If filepath provided and exists, load from file
        if self._filepath and Path(self._filepath).exists():
            with open(self._filepath, "r") as f:
                data = json.load(f)
                self._images = {
                    image_id: Image(**image_data) for image_id, image_data in data["images"].items()
                }
        else:
            # Create empty store
            self._images = {}

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
        """Add an image to the store."""
        self._images[image.id] = image

    def save(self, filepath: str | None = None) -> None:
        """Save the image store to a JSON file.

        Args:
            filepath: Path to save to. If not provided, uses the filepath from initialization.
        """
        save_path = filepath or self._filepath
        if not save_path:
            raise ValueError(
                "No filepath provided and no default filepath set during initialization"
            )

        save_path = str(save_path)
        data = {
            "images": {image_id: image.model_dump() for image_id, image in self._images.items()}
        }
        with open(save_path, "w") as f:
            json.dump(data, f)
