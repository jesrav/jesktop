"""Tests for LocalImageStore functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from jesktop.domain.image import Image
from jesktop.image_store.local import LocalImageStore


@pytest.fixture
def sample_image() -> Image:
    """Create a sample image for testing."""
    return Image(
        id="test_hash_123",
        note_id="note_456",
        content=b"fake image content",
        mime_type="image/png",
        relative_path="test_image.png",
        absolute_path="/path/to/test_image.png",
    )


@pytest.fixture
def second_image() -> Image:
    """Create a second sample image for testing."""
    return Image(
        id="test_hash_789",
        note_id="note_456",
        content=b"another fake image",
        mime_type="image/jpg",
        relative_path="another_image.jpg",
        absolute_path="/path/to/another_image.jpg",
    )


def test_empty_image_store() -> None:
    """Test that empty LocalImageStore works correctly."""
    store = LocalImageStore()

    assert store.get_image_ids() == [], "Empty store should have no image IDs"
    assert store.get_image_id_by_path("note_456", "nonexistent.png") is None, (
        "Should return None for non-existent image paths"
    )

    with pytest.raises(KeyError, match="Image nonexistent not found"):
        store.get_image("nonexistent")


def test_add_and_retrieve_image(sample_image: Image) -> None:
    """Test adding and retrieving images."""
    store = LocalImageStore()

    store.add_image(sample_image)

    retrieved_image = store.get_image("test_hash_123")
    assert retrieved_image.id == "test_hash_123", "Retrieved image should have correct ID"
    assert retrieved_image.note_id == "note_456", "Retrieved image should have correct note ID"
    assert retrieved_image.relative_path == "test_image.png", (
        "Retrieved image should have correct relative path"
    )
    assert retrieved_image.content == b"fake image content", (
        "Retrieved image should have correct content"
    )
    assert retrieved_image.mime_type == "image/png", "Retrieved image should have correct MIME type"

    assert store.get_image_ids() == ["test_hash_123"], (
        "get_image_ids should return list with single image ID"
    )


def test_get_image_id_by_path(sample_image: Image, second_image: Image) -> None:
    """Test retrieving image ID by note ID and relative path."""
    store = LocalImageStore()
    store.add_image(sample_image)
    store.add_image(second_image)

    assert store.get_image_id_by_path("note_456", "test_image.png") == "test_hash_123", (
        "Should find first image by note ID and path"
    )
    assert store.get_image_id_by_path("note_456", "another_image.jpg") == "test_hash_789", (
        "Should find second image by note ID and path"
    )

    assert store.get_image_id_by_path("wrong_note", "test_image.png") is None, (
        "Should return None for wrong note ID"
    )
    assert store.get_image_id_by_path("note_456", "wrong_path.png") is None, (
        "Should return None for wrong path"
    )


def test_multiple_images(sample_image: Image, second_image: Image) -> None:
    """Test storing multiple images."""
    store = LocalImageStore()

    store.add_image(sample_image)
    store.add_image(second_image)

    image_ids = store.get_image_ids()
    assert set(image_ids) == {
        "test_hash_123",
        "test_hash_789",
    }, "get_image_ids should return both image IDs"

    assert store.get_image("test_hash_123").relative_path == "test_image.png", (
        "First image should be retrievable with correct path"
    )
    assert store.get_image("test_hash_789").relative_path == "another_image.jpg", (
        "Second image should be retrievable with correct path"
    )


def test_save_and_load_functionality(sample_image: Image, second_image: Image) -> None:
    """Test saving to and loading from file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    Path(filepath).unlink()

    try:
        store = LocalImageStore(filepath=filepath)
        store.add_image(sample_image)
        store.add_image(second_image)

        store.save()

        assert Path(filepath).exists(), "File should be created after save"
        with open(filepath, "r") as f:
            data = json.load(f)

        assert "images" in data, "Saved data should contain images section"
        assert len(data["images"]) == 2, "Should save 2 images"
        assert "test_hash_123" in data["images"], "Should save first image"
        assert "test_hash_789" in data["images"], "Should save second image"

        new_store = LocalImageStore(filepath=filepath)

        assert set(new_store.get_image_ids()) == {
            "test_hash_123",
            "test_hash_789",
        }, "Should load both image IDs"
        assert new_store.get_image("test_hash_123").relative_path == "test_image.png", (
            "Should load first image correctly"
        )
        assert new_store.get_image("test_hash_789").relative_path == "another_image.jpg", (
            "Should load second image correctly"
        )

    finally:
        # Clean up
        if Path(filepath).exists():
            Path(filepath).unlink()


def test_save_without_filepath() -> None:
    """Test that save() raises error when no filepath is set."""
    store = LocalImageStore()

    with pytest.raises(ValueError, match="No filepath provided and no default filepath set"):
        store.save()


def test_save_with_explicit_filepath(sample_image: Image) -> None:
    """Test saving with explicit filepath parameter."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        store = LocalImageStore()
        store.add_image(sample_image)

        store.save(filepath=filepath)

        assert Path(filepath).exists(), "File should be created with explicit filepath"

        new_store = LocalImageStore(filepath=filepath)
        assert new_store.get_image_ids() == ["test_hash_123"], (
            "Should load image from explicitly saved file"
        )

    finally:
        if Path(filepath).exists():
            Path(filepath).unlink()


def test_auto_load_nonexistent_file() -> None:
    """Test that LocalImageStore handles nonexistent files gracefully."""
    nonexistent_path = "/tmp/definitely_does_not_exist_12345.json"

    store = LocalImageStore(filepath=nonexistent_path)
    assert store.get_image_ids() == [], "Should create empty store for nonexistent file"


def test_overwrite_image_with_same_id(sample_image: Image) -> None:
    """Test that adding image with same ID overwrites the previous one."""
    store = LocalImageStore()

    store.add_image(sample_image)
    assert store.get_image("test_hash_123").relative_path == "test_image.png", (
        "Original image should be stored correctly"
    )

    modified_image = Image(
        id="test_hash_123",
        note_id="different_note",
        content=b"modified content",
        mime_type="image/gif",
        relative_path="modified_path.gif",
        absolute_path="/new/path.gif",
    )

    store.add_image(modified_image)

    assert store.get_image_ids() == ["test_hash_123"], (
        "Should still have only one image with that ID"
    )

    retrieved = store.get_image("test_hash_123")
    assert retrieved.note_id == "different_note", "Image note_id should be updated"
    assert retrieved.relative_path == "modified_path.gif", "Image relative_path should be updated"
    assert retrieved.content == b"modified content", "Image content should be updated"
