"""Orchestration service for the complete ingestion pipeline."""

import logging
from hashlib import md5, sha256
from pathlib import Path

from jesktop.domain.note import EmbeddedChunk, Note
from jesktop.domain.relationships import RelationshipGraph
from jesktop.embedders.base import Embedder
from jesktop.image_store.base import ImageStore
from jesktop.vector_dbs.base import VectorDB

from .content_extractor import ContentExtractor
from .path_resolver import PathResolver
from .relationship_extraction import ReferenceResolver, RelationshipGraphBuilder
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """Orchestrates the complete ingestion pipeline from raw notes to vector database."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_db: VectorDB,
        image_store: ImageStore,
        max_tokens: int = 1000,
        overlap: int = 100,
        attachment_folders: list[str] | None = None,
    ):
        """Initialize the orchestrator with required services.

        Args:
            embedder: Embedder service for creating vector embeddings
            vector_db: Vector database for storing notes and chunks
            image_store: Image store for storing processed images
            max_tokens: Maximum tokens per text chunk
            overlap: Token overlap between chunks
            attachment_folders: List of attachment folder names to search
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.image_store = image_store
        self.attachment_folders = attachment_folders or ["Z - Attachements"]

        self.text_chunker = TextChunker(max_tokens=max_tokens, overlap=overlap)
        self.content_extractor = ContentExtractor()
        self.graph_builder = RelationshipGraphBuilder()

    def ingest(self, folder: Path) -> None:
        """Process markdown files incrementally, updating only changed files and rebuilding relationships.

        Args:
            folder: Path to folder containing markdown files
        """
        all_files = self._get_all_markdown_files_for_ingestion(folder)
        modified_files = self._get_modified_files(all_files)

        logger.info(
            f"Found {len(all_files)} total files, {len(modified_files)} modified since last ingestion"
        )

        current_note_ids = {self._generate_note_id(f, folder) for f in all_files}

        existing_note_ids = self.vector_db.get_all_note_ids()
        deleted_note_ids = existing_note_ids - current_note_ids

        if deleted_note_ids:
            logger.info(f"Deleting {len(deleted_note_ids)} removed notes...")
            for note_id in deleted_note_ids:
                self.vector_db.delete_note(note_id)

        if modified_files:
            logger.info(f"Processing {len(modified_files)} modified files...")

            notes, chunks, _ = self._process_modified_files(modified_files, folder)

            for note in notes.values():
                self.vector_db.delete_chunks_for_note(note.id)
                self.vector_db.update_note(note)

            for chunk in chunks.values():
                self.vector_db.add_chunk(chunk)

        logger.info("Rebuilding relationship graph...")
        all_notes = self.vector_db.get_notes_by_ids(list(current_note_ids))
        path_to_file_mapping = self._get_path_to_file_mapping(all_files, folder)
        relationship_graph = self._extract_and_build_relationships(all_notes, path_to_file_mapping)
        self.vector_db.update_relationship_graph(relationship_graph)

        logger.info("Ingestion complete:")
        logger.info(f"  - Total files: {len(all_files)}")
        logger.info(f"  - Modified: {len(modified_files)}")
        logger.info(f"  - Deleted: {len(deleted_note_ids)}")
        logger.info(f"  - Relationships: {len(relationship_graph.relationships)}")

        self.vector_db.save()
        self.image_store.save()

    def _process_modified_files(
        self, files: list[Path], folder: Path
    ) -> tuple[dict[str, Note], dict[str, EmbeddedChunk], dict[str, str]]:
        """Process file content, images, and metadata for modified files.

        Args:
            files: List of markdown files to process
            folder: Base folder path

        Returns:
            Tuple of (notes dict, chunks dict, note mapping)
        """
        notes = {}
        chunks = {}
        note_mapping = {}

        path_resolver = PathResolver(base_path=folder, attachment_folders=self.attachment_folders)
        note_mapping = self._get_path_to_file_mapping(files, folder)

        for file in files:
            note, note_chunks = self._process_file_content(
                file=file,
                folder=folder,
                path_resolver=path_resolver,
            )
            notes[note.id] = note
            for chunk in note_chunks:
                chunks[chunk.id] = chunk

        return notes, chunks, note_mapping

    def _extract_and_build_relationships(
        self, notes: dict[str, Note], note_mapping: dict[str, str]
    ) -> RelationshipGraph:
        """Extract relationships from note content and build relationship graph.

        Args:
            notes: Dictionary of processed notes with content
            note_mapping: Pre-built mapping from names to IDs

        Returns:
            RelationshipGraph with all relationships
        """
        for note in notes.values():
            wikilinks = self.content_extractor.extract_wikilinks(note.content)
            resolver = ReferenceResolver(note_mapping)
            note.outbound_links = resolver.resolve_references(wikilinks)

            embeds = self.content_extractor.extract_embedded_content(note.content)
            note.embedded_content = [sha256(embed.encode()).hexdigest() for embed in embeds]

        relationship_graph = self.graph_builder.build_relationships(notes)
        self.graph_builder.update_inbound_links(notes, relationship_graph.relationships)

        return relationship_graph

    def _process_file_content(
        self,
        *,
        file: Path,
        folder: Path,
        path_resolver: PathResolver,
    ) -> tuple[Note, list[EmbeddedChunk]]:
        """Process a single markdown file's content without relationships."""
        logger.debug(f"Processing {file}")

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        title = file.stem
        if content.startswith("#"):
            title = content.split("\n")[0].lstrip("#").strip()

        note_id = self._generate_note_id(file, folder)

        self.content_extractor.process_images_in_note(
            content=content,
            note_id=note_id,
            file=file,
            image_store=self.image_store,
            path_resolver=path_resolver,
        )
        self.content_extractor.process_excalidraw_refs_in_note(
            content=content,
            note_id=note_id,
            file=file,
            image_store=self.image_store,
            path_resolver=path_resolver,
        )

        content = self.content_extractor.replace_image_paths(content, note_id)

        relative_path = file.relative_to(folder)
        folder_path = str(relative_path.parent) if relative_path.parent != Path(".") else ""

        note = Note(
            id=note_id,
            title=title,
            path=str(file),
            content=content,
            created=file.stat().st_ctime,
            modified=file.stat().st_mtime,
            outbound_links=[],
            embedded_content=[],
            folder_path=folder_path,
        )

        chunks = []
        text_chunks = self.text_chunker.chunk_text(content)
        current_pos = 0

        for i, chunk_text in enumerate(text_chunks):
            start_pos = content.find(chunk_text, current_pos)
            end_pos = start_pos + len(chunk_text)
            current_pos = end_pos

            chunk = EmbeddedChunk(
                id=f"{note_id}_{i}",
                note_id=note_id,
                title=title,
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                vector=self.embedder.embed(chunk_text),
            )
            chunks.append(chunk)

        return note, chunks

    def _get_all_markdown_files_for_ingestion(self, folder: Path) -> list[Path]:
        """Get all markdown files suitable for ingestion.

        Args:
            folder: Path to folder containing markdown files

        Returns:
            List of markdown files, excluding excalidraw files
        """
        all_files = list(folder.rglob("*.md"))
        return [f for f in all_files if not f.name.endswith(".excalidraw.md")]

    def _get_modified_files(self, all_files: list[Path]) -> list[Path]:
        """Filter files for those modified since last ingestion.

        Args:
            all_files: List of all markdown files to check

        Returns:
            List of files modified since last ingestion
        """
        existing_note_ids = self.vector_db.get_all_note_ids()
        last_modified_time = 0.0
        for note_id in existing_note_ids:
            note = self.vector_db.get_note(note_id)
            if note and note.modified > last_modified_time:
                last_modified_time = note.modified

        return [f for f in all_files if f.stat().st_mtime > last_modified_time]

    @staticmethod
    def _generate_note_id(file: Path, base_folder: Path) -> str:
        """Generate a unique note ID from file path."""
        return md5(str(file.relative_to(base_folder)).encode()).hexdigest()

    @staticmethod
    def _get_path_to_file_mapping(files: list[Path], folder: Path) -> dict[str, str]:
        """Create mapping from file names/paths to IDs for reference resolution.

        Args:
            files: List of markdown files
            folder: Base folder path

        Returns:
            Dictionary mapping file names/paths to their corresponding IDs
        """
        mapping = {}

        for file in files:
            note_id = IngestionOrchestrator._generate_note_id(file, folder)
            mapping[file.stem] = note_id
            mapping[file.name] = note_id
            mapping[str(file.relative_to(folder))] = note_id

        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff"}
        for ext in image_extensions:
            for image_file in folder.rglob(f"*{ext}"):
                image_id = f"image:{image_file.relative_to(folder)}"
                mapping[image_file.stem] = image_id
                mapping[image_file.name] = image_id
                mapping[str(image_file.relative_to(folder))] = image_id

        for excalidraw_file in folder.rglob("*.excalidraw"):
            excalidraw_id = f"excalidraw:{excalidraw_file.relative_to(folder)}"
            mapping[excalidraw_file.stem] = excalidraw_id
            mapping[excalidraw_file.name] = excalidraw_id
            mapping[str(excalidraw_file.relative_to(folder))] = excalidraw_id

        return mapping
