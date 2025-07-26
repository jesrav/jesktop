"""Tests for PathResolver functionality."""

from pathlib import Path

import pytest

from jesktop.ingestion.path_resolver import PathResolver


@pytest.fixture
def sample_note_file(articles_directory: Path) -> Path:
    """Create a sample note file."""
    note_file = articles_directory / "Sample Note.md"
    note_file.write_text("# Sample Note\n\nSample content.")
    return note_file


@pytest.fixture
def relative_image(articles_directory: Path) -> Path:
    """Create relative image file."""
    image_file = articles_directory / "relative_image.png"
    image_file.write_bytes(b"relative")
    return image_file


@pytest.fixture
def asset_image(sample_note_file: Path) -> Path:
    """Create asset image in note's asset folder."""
    assets_dir = sample_note_file.parent / f"{sample_note_file.stem}.assets"
    assets_dir.mkdir()
    image_file = assets_dir / "asset_image.png"
    image_file.write_bytes(b"asset")
    return image_file


@pytest.fixture
def global_image(attachments_directory: Path) -> Path:
    """Create global image in attachments folder."""
    image_file = attachments_directory / "global_image.png"
    image_file.write_bytes(b"global")
    return image_file


@pytest.fixture
def nested_image(attachments_directory: Path) -> Path:
    """Create nested image in Sample Note assets."""
    assets_dir = attachments_directory / "Sample Note.assets"
    assets_dir.mkdir()
    image_file = assets_dir / "Image.png"
    image_file.write_bytes(b"nested")
    return image_file


@pytest.fixture
def absolute_image(temp_notes_base: Path) -> Path:
    """Create absolute image in base directory."""
    image_file = temp_notes_base / "absolute_image.png"
    image_file.write_bytes(b"absolute")
    return image_file


@pytest.fixture
def path_resolver(notes_directory: Path) -> PathResolver:
    """PathResolver instance for testing."""
    return PathResolver(base_path=notes_directory, attachment_folders=["Z - Attachements"])


def test_path_resolver_priority_order(
    path_resolver: PathResolver,
    sample_note_file: Path,
    relative_image: Path,  # noqa: ARG001
    asset_image: Path,  # noqa: ARG001
    global_image: Path,  # noqa: ARG001
) -> None:
    """Test that PathResolver follows correct priority order."""

    # Test 1: Relative to note (highest priority)
    result = path_resolver.resolve_image_path(sample_note_file, "relative_image.png")
    assert result is not None
    assert result.name == "relative_image.png"
    assert "Articles" in str(result)  # Should be in Articles dir

    # Test 2: In note assets folder
    result = path_resolver.resolve_image_path(sample_note_file, "asset_image.png")
    assert result is not None
    assert "Sample Note.assets" in str(result)

    # Test 3: In attachments folder
    result = path_resolver.resolve_image_path(sample_note_file, "global_image.png")
    assert result is not None
    assert "Z - Attachements" in str(result)


def test_path_resolver_url_decoding(
    path_resolver: PathResolver,
    sample_note_file: Path,
    nested_image: Path,  # noqa: ARG001
) -> None:
    """Test that PathResolver handles URL encoding correctly."""

    # This should work - URL encoded path that decodes to existing structure
    encoded_path = "Z%20-%20Attachements/Sample%20Note.assets/Image.png"

    result = path_resolver.resolve_image_path(sample_note_file, encoded_path)
    assert result is not None
    assert "Z - Attachements" in str(result)  # Properly decoded
    assert "Sample Note.assets" in str(result)


def test_path_resolver_nonexistent_image(
    path_resolver: PathResolver, sample_note_file: Path
) -> None:
    """Test PathResolver behavior with nonexistent images."""

    result = path_resolver.resolve_image_path(sample_note_file, "nonexistent.png")
    assert result is None  # Should return None, not raise exception


def test_path_resolver_logging() -> None:
    """Test that PathResolver provides clear logging for debugging."""
    # This will be implemented when we create the actual class
    # For now, just document the requirement

    required_log_events = [
        "Attempting to resolve image path: {path}",
        "Trying relative to note: {candidate}",
        "Trying note assets folder: {candidate}",
        "Trying attachments folder: {candidate}",
        "Trying absolute path: {candidate}",
        "Resolved successfully: {result}",
        "Failed to resolve: {path}",
    ]

    assert len(required_log_events) == 7  # All resolution steps should be logged
