#!/usr/bin/env python3
"""Test suite for relationship extraction functionality."""

from pathlib import Path

import pytest

from scripts.ingest import (
    build_note_mapping,
    calculate_relationship_strength,
    extract_embedded_content,
    extract_relationship_context,
    extract_wikilinks,
    resolve_note_references,
)


def test_link_extraction() -> None:
    """Test link extraction functions."""
    content = """
    # Test Note
    
    This note references [[Pensieve]] and [[Another Note]].
    
    It also embeds some content: ![[drawing.excalidraw]]
    
    And images: ![[image.png]]
    
    Another reference to [[Pensieve|custom text]].
    """

    wikilinks = extract_wikilinks(content)
    embeds = extract_embedded_content(content)

    assert "Pensieve" in wikilinks
    assert "Another Note" in wikilinks
    assert "drawing.excalidraw" in embeds
    assert "image.png" in embeds


def test_reference_resolution() -> None:
    """Test note reference resolution."""
    note_mapping = {
        "Pensieve": "note_id_1",
        "Another Note.md": "note_id_2",
        "Test File": "note_id_3",
    }

    links = ["Pensieve", "Another Note", "Test File", "Nonexistent"]
    resolved = resolve_note_references(links, note_mapping)

    assert "note_id_1" in resolved
    assert "note_id_2" in resolved
    assert "note_id_3" in resolved
    assert len(resolved) == 3  # Nonexistent should not be included


def test_image_reference_resolution() -> None:
    """Test image file reference resolution."""
    note_mapping = {
        "Normal Note": "note_id_1",
        "Pasted image 20250628112432.png": "image:Z - Attachements/Pasted image 20250628112432.png",
        "Pasted image 20250628112432": "image:Z - Attachements/Pasted image 20250628112432.png",
        "diagram.png": "image:images/diagram.png",
        "diagram": "image:images/diagram.png",
    }

    # Test exact image filename match
    links = ["Pasted image 20250628112432.png", "diagram.png"]
    resolved = resolve_note_references(links, note_mapping)

    assert "image:Z - Attachements/Pasted image 20250628112432.png" in resolved
    assert "image:images/diagram.png" in resolved
    assert len(resolved) == 2

    # Test image stem match
    links = ["Pasted image 20250628112432", "diagram"]
    resolved = resolve_note_references(links, note_mapping)

    assert "image:Z - Attachements/Pasted image 20250628112432.png" in resolved
    assert "image:images/diagram.png" in resolved
    assert len(resolved) == 2


def test_excalidraw_reference_resolution() -> None:
    """Test excalidraw file reference resolution."""
    note_mapping = {
        "Normal Note": "note_id_1",
        "New Org 2024-12-14 19.15.37.excalidraw": "excalidraw:drawings/New Org 2024-12-14 19.15.37.excalidraw",
        "New Org 2024-12-14 19.15.37": "excalidraw:drawings/New Org 2024-12-14 19.15.37.excalidraw",
        "workflow.excalidraw": "excalidraw:workflow.excalidraw",
        "workflow": "excalidraw:workflow.excalidraw",
    }

    # Test exact excalidraw filename match
    links = ["New Org 2024-12-14 19.15.37.excalidraw", "workflow.excalidraw"]
    resolved = resolve_note_references(links, note_mapping)

    assert "excalidraw:drawings/New Org 2024-12-14 19.15.37.excalidraw" in resolved
    assert "excalidraw:workflow.excalidraw" in resolved
    assert len(resolved) == 2

    # Test excalidraw stem match
    links = ["New Org 2024-12-14 19.15.37", "workflow"]
    resolved = resolve_note_references(links, note_mapping)

    assert "excalidraw:drawings/New Org 2024-12-14 19.15.37.excalidraw" in resolved
    assert "excalidraw:workflow.excalidraw" in resolved
    assert len(resolved) == 2


def test_mixed_reference_resolution() -> None:
    """Test resolution of mixed note, image, and excalidraw references."""
    note_mapping = {
        "Project Notes": "note_id_1",
        "image.png": "image:assets/image.png",
        "diagram.excalidraw": "excalidraw:diagrams/diagram.excalidraw",
        "Another Note.md": "note_id_2",
    }

    links = ["Project Notes", "image.png", "diagram.excalidraw", "Another Note"]
    resolved = resolve_note_references(links, note_mapping)

    assert "note_id_1" in resolved
    assert "image:assets/image.png" in resolved
    assert "excalidraw:diagrams/diagram.excalidraw" in resolved
    assert "note_id_2" in resolved
    assert len(resolved) == 4


def test_relationship_strength() -> None:
    """Test relationship strength calculation."""
    content = """
    # Main Note
    
    This mentions Pensieve once.
    
    ## Section about Pensieve
    
    Here we talk about Pensieve again and more about Pensieve.
    """

    strength = calculate_relationship_strength(content, "Pensieve")

    assert strength > 0.5  # Should be high due to header mention and frequency


def test_relationship_context() -> None:
    """Test relationship context extraction."""
    content = """
    This is some text before the reference.
    We need to discuss this with [[Important Note]] for the project.
    This is some text after the reference.
    """

    context = extract_relationship_context(content, "Important Note")

    assert "discuss this with" in context
    assert "for the project" in context


def test_note_mapping() -> None:
    """Test note mapping generation."""
    files = [
        Path("data/notes/test1.md"),
        Path("data/notes/folder/test2.md"),
    ]
    folder = Path("data/notes")

    mapping = build_note_mapping(files, folder)

    assert "test1" in mapping
    assert "test1.md" in mapping
    assert "folder/test2.md" in mapping


def test_note_mapping_with_assets() -> None:
    """Test note mapping includes image and excalidraw files."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        (temp_path / "note1.md").touch()
        (temp_path / "assets").mkdir()
        (temp_path / "assets" / "image.png").touch()
        (temp_path / "drawing.excalidraw").touch()
        (temp_path / "subdir").mkdir()
        (temp_path / "subdir" / "another.jpg").touch()

        # Test with markdown files only
        md_files = [temp_path / "note1.md"]
        mapping = build_note_mapping(md_files, temp_path)

        # Should contain markdown file mappings
        assert "note1" in mapping
        assert "note1.md" in mapping

        # Should also contain asset file mappings
        assert "image.png" in mapping
        assert "image" in mapping
        assert mapping["image.png"].startswith("image:")
        assert mapping["image"].startswith("image:")

        assert "drawing.excalidraw" in mapping
        assert "drawing" in mapping
        assert mapping["drawing.excalidraw"].startswith("excalidraw:")
        assert mapping["drawing"].startswith("excalidraw:")

        assert "another.jpg" in mapping
        assert "another" in mapping
        assert mapping["another.jpg"].startswith("image:")
        assert mapping["another"].startswith("image:")


def test_sample_notes() -> None:
    """Test with actual sample notes from the repository."""
    notes_dir = Path("data/notes")
    if not notes_dir.exists():
        pytest.skip("Notes directory not found")

    # Find a few sample files
    sample_files = list(notes_dir.rglob("*.md"))[:3]

    for file_path in sample_files:
        if file_path.name.endswith(".excalidraw.md"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        wikilinks = extract_wikilinks(content)
        embeds = extract_embedded_content(content)

        # Just verify these don't raise exceptions
        assert isinstance(wikilinks, list)
        assert isinstance(embeds, list)


def test_wikilink_patterns() -> None:
    """Test various wikilink patterns."""
    test_cases = [
        ("[[Simple Link]]", ["Simple Link"]),
        ("[[Link with spaces]]", ["Link with spaces"]),
        ("[[Link|Display Text]]", ["Link"]),
        ("Multiple [[Link1]] and [[Link2]]", ["Link1", "Link2"]),
    ]

    for content, expected in test_cases:
        wikilinks = extract_wikilinks(content)
        for exp in expected:
            assert exp in wikilinks

    # Test nested case separately to understand actual behavior
    nested_content = "Nested [[Outer [[Inner]] Link]]"
    nested_links = extract_wikilinks(nested_content)
    # Just verify it extracts something without breaking
    assert len(nested_links) > 0


def test_embed_patterns() -> None:
    """Test various embed patterns."""
    test_cases = [
        ("![[image.png]]", ["image.png"]),
        ("![[drawing.excalidraw]]", ["drawing.excalidraw"]),
        ("Multiple ![[img1.jpg]] and ![[img2.png]]", ["img1.jpg", "img2.png"]),
    ]

    for content, expected in test_cases:
        embeds = extract_embedded_content(content)
        for exp in expected:
            assert exp in embeds


def test_relationship_strength_edge_cases() -> None:
    """Test relationship strength calculation edge cases."""
    # No mentions
    strength = calculate_relationship_strength("No mentions here", "Missing")
    assert strength == 0.0

    # Single mention
    strength = calculate_relationship_strength("This mentions Target once", "Target")
    assert 0.0 < strength <= 0.3

    # Header mention
    strength = calculate_relationship_strength("# Header with Target\nSome content", "Target")
    assert strength >= 0.2  # Should get header boost


def test_relationship_context_edge_cases() -> None:
    """Test relationship context extraction edge cases."""
    # No mention
    context = extract_relationship_context("No mention here", "Missing")
    assert context == ""

    # Very short content
    context = extract_relationship_context("Short Target text", "Target")
    assert "Target" in context

    # Long content
    long_content = "A" * 200 + " Target " + "B" * 200
    context = extract_relationship_context(long_content, "Target", context_chars=50)
    assert len(context) <= 150  # Should be trimmed
    assert "Target" in context
