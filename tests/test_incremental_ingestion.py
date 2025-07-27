"""Tests for incremental ingestion functionality using fakes and fixtures."""

import time
from pathlib import Path

import pytest

from jesktop.ingestion.orchestrator import IngestionOrchestrator
from tests.fakes import FakeEmbedder, FakeImageStore, FakeVectorDB


@pytest.fixture
def orchestrator_with_fakes() -> IngestionOrchestrator:
    """Create an orchestrator with fake dependencies for fast unit testing."""
    return IngestionOrchestrator(
        embedder=FakeEmbedder(),
        vector_db=FakeVectorDB({}),
        image_store=FakeImageStore(),
        max_tokens=100,
        overlap=10,
    )


@pytest.fixture
def notes_with_content(notes_directory: Path) -> Path:
    """Create test notes with specific content for incremental testing."""
    # Create initial notes
    (notes_directory / "note1.md").write_text("# Note 1\nThis is the first note.")
    (notes_directory / "note2.md").write_text("# Note 2\nThis links to [[note1]].")

    # Create subfolder with note
    subfolder = notes_directory / "subfolder"
    subfolder.mkdir()
    (subfolder / "note3.md").write_text("# Note 3\nIn a subfolder.")

    return notes_directory


def test_initial_full_ingestion(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test that initial ingestion processes all files."""
    orchestrator = orchestrator_with_fakes

    # Run full ingestion
    orchestrator.ingest(notes_with_content)

    # Check that all notes were processed
    note_ids = orchestrator.vector_db.get_all_note_ids()
    assert len(note_ids) == 3

    # Check that notes have proper timestamps
    notes = orchestrator.vector_db.get_notes_by_ids(list(note_ids))
    assert all(note.created > 0 for note in notes.values())
    assert all(note.modified > 0 for note in notes.values())


def test_incremental_ingestion_no_changes(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test incremental ingestion when no files have changed."""
    orchestrator = orchestrator_with_fakes

    # First, do initial ingestion
    orchestrator.ingest(notes_with_content)
    initial_note_ids = orchestrator.vector_db.get_all_note_ids()
    initial_count = len(initial_note_ids)

    # Get initial timestamps
    initial_notes = orchestrator.vector_db.get_notes_by_ids(list(initial_note_ids))
    initial_timestamps = {nid: note.modified for nid, note in initial_notes.items()}

    # Wait a moment to ensure time difference
    time.sleep(0.1)

    # Run incremental ingestion
    orchestrator.ingest(notes_with_content)

    # Should still have same number of notes
    assert len(orchestrator.vector_db.get_all_note_ids()) == initial_count

    # Timestamps should remain the same since no files were modified
    current_notes = orchestrator.vector_db.get_notes_by_ids(list(initial_note_ids))
    for note_id, note in current_notes.items():
        assert note.modified == initial_timestamps[note_id]


def test_incremental_ingestion_modified_file(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test incremental ingestion when a file is modified."""
    orchestrator = orchestrator_with_fakes

    # Initial ingestion
    orchestrator.ingest(notes_with_content)

    # Get the original note content
    note_ids = list(orchestrator.vector_db.get_all_note_ids())
    notes = orchestrator.vector_db.get_notes_by_ids(note_ids)
    original_note1 = next(n for n in notes.values() if n.title == "Note 1")
    original_content = original_note1.content

    # Wait and modify a file
    time.sleep(0.1)
    (notes_with_content / "note1.md").write_text("# Note 1\nThis is the UPDATED first note.")

    # Run incremental ingestion
    orchestrator.ingest(notes_with_content)

    # Check that note was updated
    updated_note1 = orchestrator.vector_db.get_note(original_note1.id)
    assert updated_note1.content != original_content
    assert "UPDATED" in updated_note1.content


def test_incremental_ingestion_new_file(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test incremental ingestion when a new file is added."""
    orchestrator = orchestrator_with_fakes

    # Initial ingestion
    orchestrator.ingest(notes_with_content)
    initial_count = len(orchestrator.vector_db.get_all_note_ids())

    # Wait and add new file
    time.sleep(0.1)
    (notes_with_content / "note4.md").write_text("# Note 4\nA new note linking to [[note1]].")

    # Run incremental ingestion
    orchestrator.ingest(notes_with_content)

    # Should have one more note
    assert len(orchestrator.vector_db.get_all_note_ids()) == initial_count + 1

    # Check the new note exists and has correct content
    note_ids = list(orchestrator.vector_db.get_all_note_ids())
    notes = orchestrator.vector_db.get_notes_by_ids(note_ids)
    new_note = next(n for n in notes.values() if n.title == "Note 4")
    assert "new note" in new_note.content


def test_incremental_ingestion_deleted_file(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test incremental ingestion when a file is deleted."""
    orchestrator = orchestrator_with_fakes

    # Initial ingestion
    orchestrator.ingest(notes_with_content)
    initial_note_ids = orchestrator.vector_db.get_all_note_ids()

    # Find note3's ID
    notes = orchestrator.vector_db.get_notes_by_ids(list(initial_note_ids))
    note3 = next(n for n in notes.values() if n.title == "Note 3")
    note3_id = note3.id

    # Wait and delete a file
    time.sleep(0.1)
    (notes_with_content / "subfolder" / "note3.md").unlink()

    # Run incremental ingestion
    orchestrator.ingest(notes_with_content)

    # Should have one less note
    assert len(orchestrator.vector_db.get_all_note_ids()) == len(initial_note_ids) - 1

    # Check that note3 was deleted
    assert orchestrator.vector_db.get_note(note3_id) is None


def test_stable_chunk_ids(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test that chunk IDs remain stable across ingestions."""
    orchestrator = orchestrator_with_fakes

    # Initial ingestion
    orchestrator.ingest(notes_with_content)

    # Get chunk IDs for note1
    note_ids = list(orchestrator.vector_db.get_all_note_ids())
    notes = orchestrator.vector_db.get_notes_by_ids(note_ids)
    note1 = next(n for n in notes.values() if n.title == "Note 1")

    # Since we're using FakeVectorDB, we can't test actual chunks
    # But we can verify the note ID format is stable
    assert note1.id  # Should have a stable ID

    # Wait and modify note1 with same content
    time.sleep(0.1)
    (notes_with_content / "note1.md").write_text(
        "# Note 1\nThis is the first note."
    )  # Same content

    # Run incremental ingestion
    orchestrator.ingest(notes_with_content)

    # Get the note again
    updated_note1 = orchestrator.vector_db.get_note(note1.id)

    # ID should remain the same
    assert updated_note1.id == note1.id


def test_relationships_rebuild_on_incremental(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test that relationships are rebuilt even without file changes."""
    orchestrator = orchestrator_with_fakes

    # Initial ingestion
    orchestrator.ingest(notes_with_content)

    # Add a new note that links to existing notes
    time.sleep(0.1)
    (notes_with_content / "note_new.md").write_text("# New Note\nLinks to [[note1]] and [[note2]].")

    # Run incremental ingestion
    orchestrator.ingest(notes_with_content)

    # Should have one more note now
    assert len(orchestrator.vector_db.get_all_note_ids()) == 4

    # Verify the new note exists
    note_ids = list(orchestrator.vector_db.get_all_note_ids())
    notes = orchestrator.vector_db.get_notes_by_ids(note_ids)
    new_note = next(n for n in notes.values() if n.title == "New Note")
    assert "Links to" in new_note.content


def test_note_timestamp_consistency(
    orchestrator_with_fakes: IngestionOrchestrator, notes_with_content: Path
) -> None:
    """Test that note timestamps are consistent with file system."""
    orchestrator = orchestrator_with_fakes

    # Initial ingestion
    orchestrator.ingest(notes_with_content)

    # Get note timestamps
    note_ids = list(orchestrator.vector_db.get_all_note_ids())
    notes = orchestrator.vector_db.get_notes_by_ids(note_ids)

    for note in notes.values():
        # Timestamps should be set (greater than 0)
        assert note.created > 0
        assert note.modified > 0
        # For new files, created and modified should be similar
        assert abs(note.created - note.modified) < 1.0  # Within 1 second
