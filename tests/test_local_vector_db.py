"""Tests for LocalVectorDB functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jesktop.domain.note import EmbeddedChunk, Note
from jesktop.domain.relationships import NoteRelationship, RelationshipGraph
from jesktop.vector_dbs.local_db import LocalVectorDB


@pytest.fixture
def first_note() -> Note:
    """Create a sample note for testing."""
    return Note(
        id="note_123",
        title="Test Note",
        path="/path/to/test_note.md",
        content="This is test content",
        metadata={"created": 1234567890, "modified": 1234567900},
        outbound_links=["note_456"],
        inbound_links=[],
        embedded_content=[],
        folder_path="test_folder",
    )


@pytest.fixture
def second_note() -> Note:
    """Create a second sample note for testing."""
    return Note(
        id="note_456",
        title="Second Note",
        path="/path/to/second_note.md",
        content="This is more test content. It references the [[First Note]].",
        metadata={"created": 1234567800, "modified": 1234567850},
        outbound_links=[],
        inbound_links=["note_123"],
        embedded_content=[],
        folder_path="test_folder",
    )


@pytest.fixture
def first_chunk() -> EmbeddedChunk:
    """Create a sample embedded chunk for testing."""
    return EmbeddedChunk(
        id=1,
        note_id="note_123",
        title="Test Note",
        text="This is test content",
        start_pos=0,
        end_pos=20,
        vector=[0.1, 0.2, 0.3, 0.4, 0.5],
    )


@pytest.fixture
def second_chunk() -> EmbeddedChunk:
    """Create a second sample embedded chunk for testing."""
    return EmbeddedChunk(
        id=2,
        note_id="note_456",
        title="Second Note",
        text="This is more test content",
        start_pos=0,
        end_pos=27,
        vector=[0.5, 0.4, 0.3, 0.2, 0.1],
    )


@pytest.fixture
def third_chunk() -> EmbeddedChunk:
    """Create a third sample embedded chunk for testing."""
    return EmbeddedChunk(
        id=3,
        note_id="note_456",
        title="Second Note",
        text=" It references the [[First Note]].",
        start_pos=28,
        end_pos=60,
        vector=[0.2, 0.3, 0.4, 0.5, 0.6],
    )


@pytest.fixture
def sample_relationship_graph() -> RelationshipGraph:
    """Create a sample relationship graph for testing."""
    return RelationshipGraph(
        relationships=[
            NoteRelationship(
                source_note_id="note_456",
                target_note_id="note_123",
                relationship_type="wikilink",
                context="[[First Note]]",
            )
        ],
        note_clusters={"test_folder": ["note_123", "note_456"]},
    )


def test_add_and_retrieve_note(first_note: Note) -> None:
    """Test adding and retrieving notes."""
    db = LocalVectorDB()
    db.add_note(first_note)

    retrieved_note = db.get_note("note_123")
    assert retrieved_note.id == "note_123", "Retrieved note ID should match"
    assert retrieved_note.title == "Test Note", "Retrieved note title should match"
    assert retrieved_note.content == "This is test content", "Retrieved note content should match"
    assert retrieved_note.folder_path == "test_folder", "Retrieved note folder path should match"


def test_add_and_retrieve_chunk(first_chunk: EmbeddedChunk) -> None:
    """Test adding and retrieving embedded chunks."""
    db = LocalVectorDB()
    db.add_chunk(first_chunk)

    query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    closest_chunks = db.get_closest_chunks(query_vector, closest=1)

    assert len(closest_chunks) == 1, "Should return one closest chunk"
    chunk = closest_chunks[0]
    assert chunk.id == 1, "Chunk ID should match"
    assert chunk.note_id == "note_123", "Chunk should belong to the correct note"
    assert chunk.text == "This is test content", "Chunk text should match original content"


def test_find_note_by_title(first_note: Note, second_note: Note) -> None:
    """Test finding notes by title."""
    db = LocalVectorDB()
    db.add_note(first_note)
    db.add_note(second_note)

    found_note = db.find_note_by_title("Test Note")
    assert found_note.id == "note_123", "Should find first note by exact title"

    found_note = db.find_note_by_title("test note")
    assert found_note.id == "note_123", "Should find first note by case-insensitive title"

    found_note = db.find_note_by_title("Second")
    assert found_note.id == "note_456", "Should find second note by partial title match"

    assert db.find_note_by_title("Nonexistent") is None, (
        "Finding a note by title that does not exist should return None"
    )


def test_related_notes(first_note: Note, second_note: Note) -> None:
    """Test finding related notes through links."""
    db = LocalVectorDB()
    db.add_note(first_note)
    db.add_note(second_note)

    related = db.get_related_notes("note_123", max_depth=1)
    assert len(related) == 1, "Should find one related note from first note"
    assert related[0].id == "note_456", "Should find second note related to first note"

    related = db.get_related_notes("note_456", max_depth=1)
    assert len(related) == 1, "Should find one related note from second note"
    assert related[0].id == "note_123", "Should find first note related to second note"


def test_note_clusters(
    first_note: Note, second_note: Note, sample_relationship_graph: RelationshipGraph
) -> None:
    """Test note clustering functionality."""
    db = LocalVectorDB()
    db.add_note(first_note)
    db.add_note(second_note)
    db.update_relationship_graph(sample_relationship_graph)

    cluster = db.get_note_cluster("note_123")
    assert len(cluster) == 1, "Should find one note in the cluster (does not include itself)"
    assert cluster[0].id == "note_456", "Cluster should contain second note"

    cluster = db.get_note_cluster("note_456")
    assert len(cluster) == 1, "Should find one note in the cluster (does not include itself)"
    assert cluster[0].id == "note_123", "Cluster should contain first note"


def test_find_path_between_notes(first_note: Note, second_note: Note) -> None:
    """Test finding paths between notes."""
    db = LocalVectorDB()
    db.add_note(first_note)
    db.add_note(second_note)

    path = db.find_path_between_notes("note_123", "note_456")
    assert path == ["note_123", "note_456"], "Path should go from note_123 to note_456"

    path = db.find_path_between_notes("note_456", "note_123")
    assert path == ["note_456", "note_123"], "Path should go from note_456 to note_123"

    path = db.find_path_between_notes("note_123", "note_123")
    assert path == ["note_123"], "Path to self should return single note"


def test_relationship_context(sample_relationship_graph: RelationshipGraph) -> None:
    """Test getting relationship context."""
    db = LocalVectorDB()
    db.update_relationship_graph(sample_relationship_graph)

    context = db.get_relationship_context("note_456", "note_123")
    assert context == "[[First Note]]", "Should return context for existing relationship"

    context = db.get_relationship_context("note_123", "note_456")
    assert context == "", "Should return empty string for non-existent relationship"


def test_clear_functionality(first_note: Note, first_chunk: EmbeddedChunk) -> None:
    """Test clearing all data from the database."""
    db = LocalVectorDB()

    db.add_note(first_note)
    db.add_chunk(first_chunk)

    assert db.get_note("note_123") is not None, "Note should exist before clearing"
    query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert len(db.get_closest_chunks(query_vector, closest=1)) == 1, (
        "Should have one chunk before clearing"
    )

    db.clear()

    assert db.get_note("note_123") is None, "Note should be gone after clearing"
    assert len(db.get_closest_chunks(query_vector, closest=1)) == 0, (
        "Should have no chunks after clearing"
    )


def test_save_and_load_functionality(
    first_note: Note,
    second_note: Note,
    first_chunk: EmbeddedChunk,
    sample_relationship_graph: RelationshipGraph,
) -> None:
    """Test saving to and loading from file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    Path(filepath).unlink()

    try:
        db = LocalVectorDB(filepath=filepath)
        db.add_note(first_note)
        db.add_note(second_note)
        db.add_chunk(first_chunk)
        db.update_relationship_graph(sample_relationship_graph)

        db.save()

        assert Path(filepath).exists(), "File should be created after save"
        with open(filepath, "r") as f:
            data = json.load(f)

        assert "notes" in data, "Saved data should contain notes section"
        assert "chunks" in data, "Saved data should contain chunks section"
        assert "relationships" in data, "Saved data should contain relationships section"
        assert len(data["notes"]) == 2, "Should save 2 notes"
        assert len(data["chunks"]) == 1, "Should save 1 chunk"

        new_db = LocalVectorDB(filepath=filepath)

        assert new_db.get_note("note_123") is not None, "First note should be loaded"
        assert new_db.get_note("note_456") is not None, "Second note should be loaded"

        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        chunks = new_db.get_closest_chunks(query_vector, closest=1)
        assert len(chunks) == 1, "Should load 1 chunk"
        assert chunks[0].id == 1, "Loaded chunk should have correct ID"

        context = new_db.get_relationship_context("note_456", "note_123")
        assert context == "[[First Note]]", "Relationship context should be loaded correctly"

    finally:
        # Clean up
        if Path(filepath).exists():
            Path(filepath).unlink()


def test_from_data_class_method(
    first_note: Note, first_chunk: EmbeddedChunk, sample_relationship_graph: RelationshipGraph
) -> None:
    """Test creating LocalVectorDB from existing data."""
    notes = {"note_123": first_note}
    chunks = {1: first_chunk}

    db = LocalVectorDB.from_data(
        notes=notes, embedded_chunks=chunks, relationship_graph=sample_relationship_graph
    )

    assert db.get_note("note_123") is not None, "Should load note from provided data"

    query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    closest_chunks = db.get_closest_chunks(query_vector, closest=1)
    assert len(closest_chunks) == 1, "Should load chunk from provided data"

    context = db.get_relationship_context("note_456", "note_123")
    assert context == "[[First Note]]", "Should load relationship graph from provided data"


def test_save_without_filepath() -> None:
    """Test that save() raises error when no filepath is set."""
    db = LocalVectorDB()

    with pytest.raises(ValueError, match="No filepath provided and no default filepath set"):
        db.save()


def test_cosine_similarity_calculation(
    first_chunk: EmbeddedChunk, second_chunk: EmbeddedChunk
) -> None:
    """Test that cosine similarity is calculated correctly for closest chunks."""
    db = LocalVectorDB()
    db.add_chunk(first_chunk)
    db.add_chunk(second_chunk)

    query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    closest_chunks = db.get_closest_chunks(query_vector, closest=2)

    assert len(closest_chunks) == 2, "Should return both chunks"
    assert closest_chunks[0].id == 1, (
        "First chunk should have highest similarity (identical vector)"
    )
    assert closest_chunks[1].id == 2, "Second chunk should have lower similarity"

    query_vector = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    closest_chunks = db.get_closest_chunks(query_vector, closest=2)

    assert len(closest_chunks) == 2, "Should return both chunks"
    assert closest_chunks[0].id == 2, (
        "Second chunk should have highest similarity (identical vector)"
    )
    assert closest_chunks[1].id == 1, "First chunk should have lower similarity"


def test_multiple_chunks_from_same_note(
    second_chunk: EmbeddedChunk, third_chunk: EmbeddedChunk
) -> None:
    """Test handling multiple chunks from the same note."""
    db = LocalVectorDB()

    db.add_chunk(second_chunk)
    db.add_chunk(third_chunk)

    query_vector = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    closest_chunks = db.get_closest_chunks(query_vector, closest=3)

    assert len(closest_chunks) == 2, "Should return both chunks from same note"
    chunk_ids = {chunk.id for chunk in closest_chunks}
    assert chunk_ids == {2, 3}, "Should return chunks with correct IDs"

    for chunk in closest_chunks:
        assert chunk.note_id == "note_456", "Both chunks should belong to same note"
        assert chunk.title == "Second Note", "Both chunks should have same note title"

    chunk_texts = {chunk.text for chunk in closest_chunks}
    assert "This is more test content" in chunk_texts, "Should include first chunk text"
    assert " It references the [[First Note]]." in chunk_texts, "Should include second chunk text"
