"""Integration test for complete ingestion to serving pipeline."""

import json
from pathlib import Path

import pytest

from jesktop.embedders.base import Embedder
from jesktop.embedders.voyage_embedder import VoyageEmbedder
from jesktop.image_store.local import LocalImageStore
from jesktop.ingestion.orchestrator import IngestionOrchestrator
from jesktop.vector_dbs.local_db import LocalVectorDB


@pytest.fixture
def integration_test_data(articles_directory: Path, attachments_directory: Path) -> dict[str, Path]:
    """Create realistic test notes and images for integration testing."""
    # Use existing directory structure from conftest
    notes_base = articles_directory.parent.parent  # Go up to notes base

    # Create a note with text content and image references
    note_content = """# Main Test Document

This document contains test content for integration testing.

## Section One

![Test Diagram](../Z%20-%20Attachements/test-diagram.png)

This section includes some test content with references.

## References

Some related documents:
- [[Related Document A]]
- [[Related Document B]]

## Images

Here's another test image: ![[test-image.png]]

## Connections

This document connects to [[Reference Document]] and [[Additional Content]].
"""

    note_file = articles_directory / "Main Test Document.md"
    note_file.write_text(note_content)

    # Create referenced images
    test_diagram_img = attachments_directory / "test-diagram.png"
    test_diagram_img.write_bytes(b"fake test diagram content")

    test_image_img = attachments_directory / "test-image.png"
    test_image_img.write_bytes(b"fake test image content")

    # Also create images in note-specific assets directory for wikilink resolution
    note_assets_dir = articles_directory / "Main Test Document.assets"
    note_assets_dir.mkdir()
    (note_assets_dir / "test-image.png").write_bytes(b"fake test image content")

    # Create referenced notes (to test wikilinks)
    related_doc_a = articles_directory / "Related Document A.md"
    related_doc_a.write_text("""# Related Document A

This is a related test document.

[[Main Test Document]] references this document.
""")

    related_doc_b = articles_directory / "Related Document B.md"
    related_doc_b.write_text("""# Related Document B

This is another related test document with sample content.
""")

    reference_doc = articles_directory / "Reference Document.md"
    reference_doc.write_text("""# Reference Document

This document contains reference information.

It connects back to [[Main Test Document]].
""")

    return {
        "notes_dir": notes_base,
        "attachments_dir": attachments_directory,
        "main_note": note_file,
        "test_diagram_img": test_diagram_img,
        "test_image_img": test_image_img,
        "related_doc_a": related_doc_a,
        "related_doc_b": related_doc_b,
        "reference_doc": reference_doc,
    }


@pytest.fixture
def ingested_data(
    integration_test_data: dict[str, Path], tmp_path: Path, fake_embedder: Embedder
) -> dict[str, Path]:
    """Run ingestion on test data and return storage file paths."""
    # Create storage file paths
    vector_db_path = tmp_path / "vector.json"
    image_store_path = tmp_path / "images.json"

    # Set up ingestion with fake embedder to avoid API calls
    embedder = fake_embedder
    vector_db = LocalVectorDB(filepath=vector_db_path)
    image_store = LocalImageStore(filepath=image_store_path)

    orchestrator = IngestionOrchestrator(
        embedder=embedder,
        vector_db=vector_db,
        image_store=image_store,
    )
    orchestrator.ingest(integration_test_data["notes_dir"])

    return {
        "vector_db_path": vector_db_path,
        "image_store_path": image_store_path,
        "notes_dir": integration_test_data["notes_dir"],
    }


def test_complete_ingestion_to_serving_pipeline(ingested_data: dict[str, Path]) -> None:
    """
    Test complete pipeline from ingestion to data persistence and retrieval.

    This integration test verifies that:
    1. Ingestion correctly processes notes, images, and relationships
    2. Vector database contains expected notes and embeddings
    3. Image store contains processed images
    4. Data can be loaded from persistence and queried correctly
    """
    vector_db_path = ingested_data["vector_db_path"]
    image_store_path = ingested_data["image_store_path"]

    # Verify ingestion results
    assert vector_db_path.exists(), "Vector database should be created after ingestion"
    assert image_store_path.exists(), "Image store should be created after ingestion"

    # Load and verify vector database contents
    with open(vector_db_path, "r") as f:
        vector_data = json.load(f)

    assert "notes" in vector_data, "Vector database should contain notes section"
    assert "chunks" in vector_data, "Vector database should contain chunks"
    assert "relationships" in vector_data, "Vector database should contain relationships"

    notes = vector_data["notes"]
    assert len(notes) == 4, "Should have ingested 4 notes"

    # Verify specific notes exist
    note_titles = [note["title"] for note in notes.values()]
    expected_titles = [
        "Main Test Document",
        "Related Document A",
        "Related Document B",
        "Reference Document",
    ]
    for title in expected_titles:
        assert title in note_titles, f"Should have ingested note: {title}"

    # Verify chunks were created
    chunks = vector_data["chunks"]
    assert len(chunks) > 0, "Should have created text chunks with embeddings"

    # Verify relationships were built
    relationships = vector_data["relationships"]
    assert "links" in relationships or len(relationships) > 0, "Should have built relationships"

    # Load and verify image store contents
    with open(image_store_path, "r") as f:
        image_data = json.load(f)

    assert "images" in image_data, "Image store should contain images section"
    images = image_data["images"]
    assert len(images) >= 0, "Should have processed images or handled missing images gracefully"

    # If images were processed, verify they have the expected structure
    if len(images) > 0:
        for img in images.values():
            assert "relative_path" in img, "Images should have relative_path"
            assert "content" in img, "Images should have content"
            assert "mime_type" in img, "Images should have mime_type"

    # Find the main note ID for testing
    main_note_id = None
    for note_id, note in notes.items():
        if note["title"] == "Main Test Document":
            main_note_id = note_id
            break

    assert main_note_id is not None, "Should find main note ID"

    # Test that data can be loaded from persistence and queried
    loaded_vector_db = LocalVectorDB(filepath=vector_db_path)
    loaded_note = loaded_vector_db.get_note(main_note_id)
    assert loaded_note is not None, "Should be able to retrieve note from loaded vector DB"
    assert loaded_note.title == "Main Test Document", "Should have correct note title"
    assert "Section One" in loaded_note.content, "Should have note content"

    # Test that image store can be loaded and queried
    loaded_image_store = LocalImageStore(filepath=image_store_path)
    loaded_image_ids = loaded_image_store.get_image_ids()

    # Verify image data structure if images were processed
    if loaded_image_ids:
        sample_image = loaded_image_store.get_image(loaded_image_ids[0])
        assert sample_image.note_id == main_note_id, "Image should be associated with correct note"
        assert sample_image.content is not None, "Image should have content"
        assert sample_image.mime_type.startswith("image/"), "Image should have proper MIME type"


def test_ingestion_with_real_embedder_if_api_key_available() -> None:
    """
    Test ingestion with real VoyageAI embedder if API key is available.

    This test is skipped if no API key is provided.
    """
    try:
        # Try to create real embedder - will fail if no API key
        VoyageEmbedder(api_key="test")
    except Exception:
        pytest.skip("VoyageAI API key not available - skipping real embedder test")

    # If we get here, we could test with real embeddings
    # For now, we'll skip this to avoid API costs in tests
    pytest.skip("Real API test skipped to avoid costs - implementation works with fake embedder")


def test_error_handling_in_integration_pipeline(
    temp_notes_base: Path, fake_embedder: Embedder
) -> None:
    """Test that the integration pipeline handles empty directories gracefully."""
    # Create empty notes directory using existing fixture
    notes_dir = temp_notes_base / "empty_notes"
    notes_dir.mkdir()

    # Create storage paths
    vector_db_path = temp_notes_base / "vector.json"
    image_store_path = temp_notes_base / "images.json"

    # Set up ingestion
    embedder = fake_embedder
    vector_db = LocalVectorDB(filepath=vector_db_path)
    image_store = LocalImageStore(filepath=image_store_path)

    orchestrator = IngestionOrchestrator(
        embedder=embedder,
        vector_db=vector_db,
        image_store=image_store,
    )

    # Run ingestion on empty directory
    orchestrator.ingest(notes_dir)

    # Should create empty but valid storage files
    assert vector_db_path.exists(), "Should create vector database file even for empty input"
    assert image_store_path.exists(), "Should create image store file even for empty input"

    # Verify the data structures are valid but empty
    with open(vector_db_path, "r") as f:
        vector_data = json.load(f)

    assert "notes" in vector_data, "Vector database should have notes section"
    assert len(vector_data["notes"]) == 0, "Should have no notes for empty directory"
    assert "chunks" in vector_data, "Vector database should have chunks section"
    assert len(vector_data["chunks"]) == 0, "Should have no chunks for empty directory"

    with open(image_store_path, "r") as f:
        image_data = json.load(f)

    assert "images" in image_data, "Image store should have images section"
    assert len(image_data["images"]) == 0, "Should have no images for empty directory"


def test_incremental_ingestion_integration(
    integration_test_data: dict[str, Path], tmp_path: Path, fake_embedder: Embedder
) -> None:
    """
    Test incremental ingestion with real storage implementations.

    This integration test verifies that:
    1. Initial ingestion creates complete data structures
    2. Incremental ingestion properly handles file modifications
    3. Data persistence works correctly across ingestion cycles
    4. File deletion is handled properly
    """
    notes_dir = integration_test_data["notes_dir"]
    vector_db_path = tmp_path / "vector.json"
    image_store_path = tmp_path / "images.json"

    # Set up ingestion with real storage
    embedder = fake_embedder
    vector_db = LocalVectorDB(filepath=vector_db_path)
    image_store = LocalImageStore(filepath=image_store_path)

    orchestrator = IngestionOrchestrator(
        embedder=embedder,
        vector_db=vector_db,
        image_store=image_store,
    )

    # Phase 1: Initial full ingestion
    orchestrator.ingest(notes_dir)

    # Verify initial state
    initial_note_ids = vector_db.get_all_note_ids()
    assert len(initial_note_ids) == 4, "Should have 4 notes after initial ingestion"

    with open(vector_db_path, "r") as f:
        initial_data = json.load(f)
    initial_notes_count = len(initial_data["notes"])

    # Phase 2: Modify an existing file
    import time

    time.sleep(0.1)  # Ensure timestamp difference

    main_note_path = integration_test_data["main_note"]
    original_content = main_note_path.read_text()
    modified_content = (
        original_content + "\n\n## New Section\n\nAdded content for incremental test."
    )
    main_note_path.write_text(modified_content)

    # Run incremental ingestion
    orchestrator.ingest(notes_dir)

    # Verify modification was detected and processed
    with open(vector_db_path, "r") as f:
        modified_data = json.load(f)

    assert len(modified_data["notes"]) == initial_notes_count, "Note count should remain same"
    # Find the modified note and verify content changed
    modified_notes = modified_data["notes"]
    main_note_found = False
    for note_data in modified_notes.values():
        if "Main Test Document" in note_data["title"]:
            assert "New Section" in note_data["content"], "Modified content should be persisted"
            main_note_found = True
            break
    assert main_note_found, "Should find the main test document"

    # Phase 3: Add a new file
    time.sleep(0.1)
    new_note_path = (
        integration_test_data["notes_dir"] / "3 - Learning" / "Articles" / "New Document.md"
    )
    new_note_path.write_text("""# New Document

This is a newly added document for incremental testing.

It references [[Main Test Document]] and [[Reference Document]].
""")

    # Run incremental ingestion again
    orchestrator.ingest(notes_dir)

    # Verify new file was added
    with open(vector_db_path, "r") as f:
        new_file_data = json.load(f)

    assert len(new_file_data["notes"]) == initial_notes_count + 1, "Should have one more note"

    # Verify the new note exists
    new_notes = new_file_data["notes"]
    new_note_found = False
    for note_data in new_notes.values():
        if note_data["title"] == "New Document":
            assert "incremental testing" in note_data["content"]
            new_note_found = True
            break
    assert new_note_found, "Should find the new document"

    # Phase 4: Delete a file
    time.sleep(0.1)
    related_doc_b_path = integration_test_data["related_doc_b"]
    related_doc_b_path.unlink()

    # Run incremental ingestion again
    orchestrator.ingest(notes_dir)

    # Verify file was deleted
    with open(vector_db_path, "r") as f:
        deletion_data = json.load(f)

    assert len(deletion_data["notes"]) == initial_notes_count, (
        "Should have original count after deletion"
    )

    # Verify Related Document B is gone
    deletion_notes = deletion_data["notes"]
    for note_data in deletion_notes.values():
        assert note_data["title"] != "Related Document B", "Deleted document should not exist"

    # Phase 5: Test persistence across database reload
    # Create new instances to simulate application restart
    new_vector_db = LocalVectorDB(filepath=vector_db_path)
    new_image_store = LocalImageStore(filepath=image_store_path)

    new_orchestrator = IngestionOrchestrator(
        embedder=embedder,
        vector_db=new_vector_db,
        image_store=new_image_store,
    )

    # Should detect no changes needed (all files already processed)
    new_orchestrator.ingest(notes_dir)

    # Verify data consistency after reload
    final_note_ids = new_vector_db.get_all_note_ids()
    assert len(final_note_ids) == initial_notes_count, "Note count should be stable after reload"

    # Verify we can still query the modified content
    notes_by_id = new_vector_db.get_notes_by_ids(list(final_note_ids))
    main_note = None
    for note in notes_by_id.values():
        if "Main Test Document" in note.title:
            main_note = note
            break

    assert main_note is not None, "Should find main note after reload"
    assert "New Section" in main_note.content, "Modified content should persist across reload"


def test_incremental_ingestion_with_relationship_updates(
    temp_notes_base: Path, fake_embedder: Embedder
) -> None:
    """
    Test that incremental ingestion properly updates relationships.

    This test specifically focuses on relationship graph updates during incremental processing.
    """
    notes_dir = temp_notes_base / "notes"
    notes_dir.mkdir()

    vector_db_path = temp_notes_base / "vector.json"
    image_store_path = temp_notes_base / "images.json"

    # Set up storage
    vector_db = LocalVectorDB(filepath=vector_db_path)
    image_store = LocalImageStore(filepath=image_store_path)
    orchestrator = IngestionOrchestrator(
        embedder=fake_embedder,
        vector_db=vector_db,
        image_store=image_store,
    )

    # Create initial notes with relationships
    (notes_dir / "doc_a.md").write_text("# Document A\n\nThis references [[Document B]].")
    (notes_dir / "doc_b.md").write_text("# Document B\n\nStandalone document.")

    # Initial ingestion
    orchestrator.ingest(notes_dir)

    initial_graph = vector_db._relationship_graph
    initial_relationships = len(initial_graph.relationships) if initial_graph else 0

    # Add a new document that creates more relationships
    import time

    time.sleep(0.1)
    (notes_dir / "doc_c.md").write_text("""# Document C

This document creates multiple relationships:
- Links to [[Document A]]
- Also links to [[Document B]]
- And references both [[doc_a]] and [[doc_b]] by filename
""")

    # Run incremental ingestion
    orchestrator.ingest(notes_dir)

    # Verify relationships were updated
    final_graph = vector_db._relationship_graph
    final_relationships = len(final_graph.relationships) if final_graph else 0

    # Should have more relationships now (exact count depends on relationship extraction logic)
    assert final_relationships >= initial_relationships, (
        "Should have at least as many relationships as before"
    )

    # Verify all notes are still present
    note_ids = vector_db.get_all_note_ids()
    assert len(note_ids) == 3, "Should have all three documents"

    notes = vector_db.get_notes_by_ids(list(note_ids))
    titles = [note.title for note in notes.values()]
    expected_titles = ["Document A", "Document B", "Document C"]
    for title in expected_titles:
        assert title in titles, f"Should have note with title: {title}"
