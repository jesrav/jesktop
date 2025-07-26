#!/usr/bin/env python3
"""Test suite for wikilink functionality."""

from jesktop.domain.note import Note
from jesktop.vector_dbs.local_db import LocalVectorDB


def test_find_note_by_title_exact_match() -> None:
    """Test finding note by exact title match."""
    notes = {
        "note1": Note(
            id="note1",
            title="My Important Note",
            path="/path/to/my_important_note.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
        "note2": Note(
            id="note2",
            title="Another Note",
            path="/path/to/another_note.md",
            content="More content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Test exact title match
    result = db.find_note_by_title("My Important Note")
    assert result is not None
    assert result.id == "note1"

    # Test case insensitive
    result = db.find_note_by_title("my important note")
    assert result is not None
    assert result.id == "note1"


def test_find_note_by_title_stem_match() -> None:
    """Test finding note by stem match."""
    notes = {
        "note1": Note(
            id="note1",
            title="My Important Note",
            path="/path/to/my_important_note.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Test stem match
    result = db.find_note_by_title("my_important_note")
    assert result is not None
    assert result.id == "note1"


def test_find_note_by_title_normalized_match() -> None:
    """Test finding note with space/underscore normalization."""
    notes = {
        "note1": Note(
            id="note1",
            title="My Important Note",
            path="/path/to/my_important_note.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Test that "My Important Note" matches when searching for "My_Important_Note"
    result = db.find_note_by_title("My_Important_Note")
    assert result is not None
    assert result.id == "note1"

    # Test reverse - underscore title matches space query
    notes2 = {
        "note2": Note(
            id="note2",
            title="My_Important_Note",
            path="/path/to/my_important_note.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db2 = LocalVectorDB.from_data(notes=notes2)
    result = db2.find_note_by_title("My Important Note")
    assert result is not None
    assert result.id == "note2"


def test_find_note_by_title_substring_match() -> None:
    """Test finding note by substring match."""
    notes = {
        "note1": Note(
            id="note1",
            title="My Important Note About Machine Learning",
            path="/path/to/my_important_note_about_machine_learning.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Test substring match
    result = db.find_note_by_title("Important Note")
    assert result is not None
    assert result.id == "note1"


def test_find_note_by_title_no_match() -> None:
    """Test finding note when no match exists."""
    notes = {
        "note1": Note(
            id="note1",
            title="My Important Note",
            path="/path/to/my_important_note.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Test no match
    result = db.find_note_by_title("Nonexistent Note")
    assert result is None


def test_find_note_by_title_empty_title() -> None:
    """Test finding note when note has no title."""
    notes = {
        "note1": Note(
            id="note1",
            title="",
            path="/path/to/untitled_note.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Should match by stem
    result = db.find_note_by_title("untitled_note")
    assert result is not None
    assert result.id == "note1"

    # Should also match empty title
    result = db.find_note_by_title("")
    assert result is not None
    assert result.id == "note1"


def test_find_note_by_title_priority_order() -> None:
    """Test that matching follows the correct priority order."""
    notes = {
        "note1": Note(
            id="note1",
            title="Test Note",
            path="/path/to/different_stem.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
        "note2": Note(
            id="note2",
            title="Longer Test Note Title",
            path="/path/to/test_note.md",
            content="More content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Should prioritize exact title match over stem or substring
    result = db.find_note_by_title("Test Note")
    assert result is not None
    assert result.id == "note1"  # Exact title match wins


def test_find_note_by_title_edge_cases() -> None:
    """Test edge cases for note finding."""
    notes = {
        "note1": Note(
            id="note1",
            title="",  # Empty title
            path="/path/to/empty_title.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
        "note2": Note(
            id="note2",
            title="  Whitespace Title  ",
            path="/path/to/whitespace_title.md",
            content="More content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Test empty string search - should find the note with empty title
    result = db.find_note_by_title("")
    assert result is not None
    assert result.id == "note1"

    # Test whitespace handling
    result = db.find_note_by_title("Whitespace Title")
    assert result is not None
    assert result.id == "note2"

    # Test searching by stem when title is empty
    result = db.find_note_by_title("empty_title")
    assert result is not None
    assert result.id == "note1"


def test_find_note_by_title_multiple_candidates() -> None:
    """Test behavior when multiple notes could match."""
    notes = {
        "note1": Note(
            id="note1",
            title="Project Notes",
            path="/path/to/project_notes.md",
            content="Some content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
        "note2": Note(
            id="note2",
            title="Important Project Notes Draft",
            path="/path/to/important_project_notes_draft.md",
            content="More content",
            metadata={},
            outbound_links=[],
            inbound_links=[],
        ),
    }

    db = LocalVectorDB.from_data(notes=notes)

    # Should return first exact match found
    result = db.find_note_by_title("Project Notes")
    assert result is not None
    assert result.id == "note1"  # Exact match should win

    # For substring, should return first match found
    result = db.find_note_by_title("Project")
    assert result is not None
    # Could be either note1 or note2, but should be consistent
