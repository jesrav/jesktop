import base64
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from jesktop.api import create_app
from jesktop.domain.image import Image
from jesktop.domain.note import Note
from jesktop.embedders.base import Embedder
from jesktop.image_store.base import ImageStore
from jesktop.ingestion.path_resolver import PathResolver
from jesktop.llms.base import LLMChat
from jesktop.vector_dbs.base import VectorDB
from tests.fakes import FakeEmbedder, FakeImageStore, FakeLLMChat, FakeVectorDB


@pytest.fixture
def test_notes() -> dict[str, Note]:
    return {
        "note1": Note(
            id="note1",
            title="Test Note 1",
            content="Test content 1",
            path="/test/note1.md",
            created=0.0,
            modified=0.0,
        ),
        "note2": Note(
            id="note2",
            title="Test Note 2",
            content="Test content 2",
            path="/test/note2.md",
            created=0.0,
            modified=0.0,
        ),
    }


@pytest.fixture
def test_images() -> dict[str, Image]:
    return {
        "image1": Image(
            id="image1",
            note_id="note1",
            content=base64.b64encode(b"fake image data"),
            mime_type="image/png",
            relative_path="test.png",
            absolute_path="/test/test.png",
        ),
        "image2": Image(
            id="image2",
            note_id="note2",
            content=base64.b64encode(b"another fake image"),
            mime_type="image/jpeg",
            relative_path="test.jpg",
            absolute_path="/test/test.jpg",
        ),
    }


@pytest.fixture
def fake_chat() -> LLMChat:
    return FakeLLMChat(
        responses=[
            "Summary\nThe notes are about emojis.",
            "Details\nIn your note titled\nThe banana emoji I found ...",
            "Additional Context\nThe notes also refer to\nresources for emojis",
        ]
    )


@pytest.fixture
def fake_embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def fake_vector_db(test_notes: dict[str, Note]) -> VectorDB:
    return FakeVectorDB(test_notes)


@pytest.fixture
def fake_image_store(test_images: dict[str, Image]) -> ImageStore:
    return FakeImageStore(test_images)


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override settings for testing."""
    monkeypatch.setattr("app.settings.auth_username", "admin")
    monkeypatch.setattr("app.settings.auth_password", "password")


@pytest.fixture
def test_client(
    fake_embedder: Embedder,
    fake_vector_db: VectorDB,
    fake_chat: LLMChat,
    fake_image_store: ImageStore,
) -> TestClient:
    """Create test client with fake implementations."""
    app = create_app(
        vector_db=fake_vector_db,
        embedder=fake_embedder,
        chatbot=fake_chat,
        image_store=fake_image_store,
    )
    return TestClient(app)


@pytest.fixture
def temp_notes_base() -> Generator[Path, None, None]:
    """Create a temporary notes directory structure
    used when testing the ingestion and parsing of notes.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def notes_directory(temp_notes_base: Path) -> Path:
    """Create notes subdirectory."""
    notes_dir = temp_notes_base / "notes"
    notes_dir.mkdir()
    return notes_dir


@pytest.fixture
def articles_directory(notes_directory: Path) -> Path:
    """Create articles subdirectory within notes."""
    articles_dir = notes_directory / "3 - Learning" / "Articles"
    articles_dir.mkdir(parents=True)
    return articles_dir


@pytest.fixture
def attachments_directory(notes_directory: Path) -> Path:
    """Create attachments directory."""
    attachments_dir = notes_directory / "Z - Attachements"
    attachments_dir.mkdir()
    return attachments_dir


@pytest.fixture
def path_resolver(notes_directory: Path) -> PathResolver:
    """PathResolver initialised using the notes directory fixture."""
    return PathResolver(base_path=notes_directory, attachment_folders=["Z - Attachements"])
