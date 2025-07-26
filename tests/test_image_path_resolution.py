"""Test image path resolution functionality."""

from pathlib import Path
from typing import Any

import pytest

from jesktop.image_store.local import LocalImageStore
from jesktop.ingestion.content_extractor import ContentExtractor
from jesktop.ingestion.path_resolver import PathResolver


@pytest.fixture
def test_note_file(articles_directory: Path) -> Path:
    """Create a test note file."""
    note_file = articles_directory / "Test Note.md"
    note_file.write_text("# Test Note\n\nTest content.")
    return note_file


@pytest.fixture
def test_note_assets_directory(attachments_directory: Path) -> Path:
    """Create Test Note assets directory."""
    assets_dir = attachments_directory / "Test Note.assets"
    assets_dir.mkdir()
    return assets_dir


@pytest.fixture
def url_encoded_test_image(test_note_assets_directory: Path) -> Path:
    """Create test image for URL-encoded path test."""
    image_file = test_note_assets_directory / "Image.png"
    image_file.write_bytes(b"test image content")
    return image_file


@pytest.fixture
def url_encoded_note_content() -> str:
    """Note content with URL-encoded image path."""
    return """# Test Note

This image has a URL-encoded path:

![Image.png](Z%20-%20Attachements/Test%20Note.assets/Image.png)
"""


def test_url_encoded_path_resolution(
    test_note_file: Path,
    url_encoded_note_content: str,
    url_encoded_test_image: Path,
    path_resolver: PathResolver,
    monkeypatch: Any,
) -> None:
    """
    Test that image ingestion correctly handles URL-encoded paths.

    The system should decode URL-encoded paths like "Z%20-%20Attachements"
    to "Z - Attachements" and successfully find the referenced images.
    """
    temp_path = test_note_file.parent.parent.parent  # Go up to notes base
    note_file = test_note_file
    note_content = url_encoded_note_content
    image_file = url_encoded_test_image

    # Verify our test setup is correct
    assert image_file.exists(), "Test image file should exist"
    assert "Z%20-%20Attachements" in note_content, "Note should contain URL-encoded path"

    # Test with PathResolver
    image_store = LocalImageStore()
    note_id = "test_note_id"

    # Change to temp directory so relative paths work
    monkeypatch.chdir(temp_path)

    # This should now successfully resolve URL-encoded paths
    content_extractor = ContentExtractor()
    content_extractor.process_images_in_note(
        content=note_content,
        note_id=note_id,
        file=note_file,
        image_store=image_store,
        path_resolver=path_resolver,
    )

    stored_image_ids = image_store.get_image_ids()

    # System should successfully resolve URL-encoded path and store the image
    assert len(stored_image_ids) == 1, (
        "URL-encoded path should be resolved correctly:"
        f" {len(stored_image_ids)} images stored, expected 1"
    )

    stored_image = image_store.get_image(stored_image_ids[0])
    assert stored_image.content == b"test image content", (
        "Stored image content should match original test image content"
    )


@pytest.fixture
def multi_pattern_note_file(articles_directory: Path) -> Path:
    """Create note file with multiple image reference patterns."""
    note_file = articles_directory / "Multi Pattern Test.md"
    note_content = """# Multi Pattern Test

1. Simple wikilink: ![[simple.png]]
2. URL encoded path: ![encoded.png](Z%20-%20Attachements/encoded.png)  
3. Relative path: ![relative.png](relative.png)
4. Asset folder: ![asset.png](asset.png)
"""
    note_file.write_text(note_content)
    return note_file


@pytest.fixture
def multi_pattern_note_content() -> str:
    """Note content with multiple image reference patterns."""
    return """# Multi Pattern Test

1. Simple wikilink: ![[simple.png]]
2. URL encoded path: ![encoded.png](Z%20-%20Attachements/encoded.png)  
3. Relative path: ![relative.png](relative.png)
4. Asset folder: ![asset.png](asset.png)
"""


@pytest.fixture
def multi_pattern_images(articles_directory: Path, attachments_directory: Path) -> dict[str, bytes]:
    """Create image files for multiple pattern test."""
    # Create assets directory for this note
    assets_dir = articles_directory / "Multi Pattern Test.assets"
    assets_dir.mkdir()

    # Create image files in various locations
    simple_img = attachments_directory / "simple.png"
    encoded_img = attachments_directory / "encoded.png"
    relative_img = articles_directory / "relative.png"
    asset_img = assets_dir / "asset.png"

    simple_img.write_bytes(b"simple content")
    encoded_img.write_bytes(b"encoded content")
    relative_img.write_bytes(b"relative content")
    asset_img.write_bytes(b"asset content")

    return {
        "simple.png": b"simple content",
        "encoded.png": b"encoded content",
        "relative.png": b"relative content",
        "asset.png": b"asset content",
    }


def test_multiple_image_reference_patterns(
    multi_pattern_note_file: Path,
    multi_pattern_note_content: str,
    multi_pattern_images: dict[str, Path],
    path_resolver: PathResolver,
    monkeypatch: Any,
) -> None:
    """
    Test that the system correctly handles various image reference patterns.

    The system should support:
    - Wikilink syntax: ![[image.png]]
    - URL-encoded paths: ![](Z%20-%20Attachements/image.png)
    - Relative paths: ![](image.png)
    - Asset folder references: ![](image.png) -> Note.assets/image.png
    """
    temp_path = multi_pattern_note_file.parent.parent.parent  # Go up to notes base
    note_file = multi_pattern_note_file
    note_content = multi_pattern_note_content
    expected_images = multi_pattern_images

    image_store = LocalImageStore()
    note_id = "multi_test"

    monkeypatch.chdir(temp_path)

    content_extractor = ContentExtractor()
    content_extractor.process_images_in_note(
        content=note_content,
        note_id=note_id,
        file=note_file,
        image_store=image_store,
        path_resolver=path_resolver,
    )

    stored_image_ids = image_store.get_image_ids()

    # System should handle all common image reference patterns
    assert len(stored_image_ids) == len(expected_images), (
        f"Should resolve all {len(expected_images)} image patterns, "
        f"but only resolved {len(stored_image_ids)}. "
        f"Expected: {list(expected_images.keys())}, "
        f"Got: {[image_store.get_image(img_id).relative_path for img_id in stored_image_ids]}"
    )
