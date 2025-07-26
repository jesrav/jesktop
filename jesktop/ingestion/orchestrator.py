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

        # Initialize services
        self.text_chunker = TextChunker(max_tokens=max_tokens, overlap=overlap)
        self.content_extractor = ContentExtractor()
        self.graph_builder = RelationshipGraphBuilder()

    def ingest_folder(self, folder: Path) -> None:
        """Process all markdown files in a folder and save to vector database and image store.

        This method uses a two-pass approach:
        1. First pass: Process file content, images, extract metadata, and build note mapping
        2. Second pass: Extract relationships and build graph using already-loaded content

        Args:
            folder: Path to folder containing markdown files
        """
        files = list(folder.rglob("*.md"))
        files = [f for f in files if not f.name.endswith(".excalidraw.md")]

        print(f"Starting ingestion of {len(files)} files...")

        # PASS 1: Content processing, metadata extraction, and mapping building
        print("Pass 1: Processing content and building note mapping...")
        notes, chunks, note_mapping = self._first_pass_content_processing(files, folder)

        # PASS 2: Relationship extraction and graph building (no file re-reading)
        print("Pass 2: Extracting relationships and building graph...")
        relationship_graph = self._second_pass_relationship_building(notes, note_mapping)

        # Update vector database with all processed data
        print("Updating vector database...")
        self._update_vector_database(notes, chunks, relationship_graph)

        print(f"Processed {len(notes)} notes into {len(chunks)} chunks")
        print(f"Found {len(relationship_graph.relationships)} relationships")
        print(f"Created {len(relationship_graph.note_clusters)} folder clusters")

        self.vector_db.save()
        self.image_store.save()

    def _first_pass_content_processing(
        self, files: list[Path], folder: Path
    ) -> tuple[dict[str, Note], dict[int, EmbeddedChunk], dict[str, str]]:
        """First pass: Process file content, images, extract metadata, and build note mapping.

        Args:
            files: List of markdown files to process
            folder: Base folder path

        Returns:
            Tuple of (notes dict, chunks dict, note mapping)
        """
        notes = {}
        chunks = {}
        note_mapping = {}
        chunk_id = 0

        # Create PathResolver for image resolution
        path_resolver = PathResolver(base_path=folder, attachment_folders=self.attachment_folders)

        # First, build the complete note mapping (including assets)
        note_mapping = self._build_complete_mapping(files, folder)

        for file in files:
            note, note_chunks = self._process_file_content(
                file=file,
                folder=folder,
                chunk_id=chunk_id,
                path_resolver=path_resolver,
            )
            notes[note.id] = note
            for chunk in note_chunks:
                chunks[chunk.id] = chunk
            chunk_id += len(note_chunks)

        return notes, chunks, note_mapping

    def _second_pass_relationship_building(
        self, notes: dict[str, Note], note_mapping: dict[str, str]
    ) -> RelationshipGraph:
        """Second pass: Extract relationships and build graph using already-loaded content.

        Args:
            notes: Dictionary of processed notes with content
            note_mapping: Pre-built mapping from names to IDs

        Returns:
            RelationshipGraph with all relationships
        """
        # Extract relationships for each note using already-loaded content
        for note in notes.values():
            # Extract wikilinks and resolve them to note IDs
            wikilinks = self.content_extractor.extract_wikilinks(note.content)
            resolver = ReferenceResolver(note_mapping)
            note.outbound_links = resolver.resolve_references(wikilinks)

            # Extract embedded content and hash them
            embeds = self.content_extractor.extract_embedded_content(note.content)
            note.embedded_content = [sha256(embed.encode()).hexdigest() for embed in embeds]

        # Build relationship graph
        relationship_graph = self.graph_builder.build_relationships(notes)

        # Update inbound links
        self.graph_builder.update_inbound_links(notes, relationship_graph.relationships)

        return relationship_graph

    def _update_vector_database(
        self,
        notes: dict[str, Note],
        chunks: dict[int, EmbeddedChunk],
        relationship_graph: RelationshipGraph,
    ) -> None:
        """Update the vector database with processed data.

        Args:
            notes: Dictionary of processed notes
            chunks: Dictionary of processed chunks
            relationship_graph: Built relationship graph
        """
        self.vector_db.clear()  # Clear any existing data
        for note in notes.values():
            self.vector_db.add_note(note)
        for chunk in chunks.values():
            self.vector_db.add_chunk(chunk)
        self.vector_db.update_relationship_graph(relationship_graph)

    def _process_file_content(
        self,
        *,
        file: Path,
        folder: Path,
        chunk_id: int,
        path_resolver: PathResolver,
    ) -> tuple[Note, list[EmbeddedChunk]]:
        """Process a single markdown file's content without relationships."""
        print(f"Processing {file}")

        # Read the file content
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract title from filename or first header
        title = file.stem
        if content.startswith("#"):
            title = content.split("\n")[0].lstrip("#").strip()

        # Generate note ID from path
        note_id = md5(str(file.relative_to(folder)).encode()).hexdigest()

        # Process images and excalidraw refs
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

        # Extract folder path for hierarchical relationships
        relative_path = file.relative_to(folder)
        folder_path = str(relative_path.parent) if relative_path.parent != Path(".") else ""

        note = Note(
            id=note_id,
            title=title,
            path=str(file),
            content=content,
            metadata={
                "created": file.stat().st_ctime,
                "modified": file.stat().st_mtime,
            },
            outbound_links=[],  # Will be populated in second pass
            embedded_content=[],  # Will be populated in second pass
            folder_path=folder_path,
        )

        # Split the text into chunks
        chunks = []
        text_chunks = self.text_chunker.chunk_text(content)
        current_pos = 0

        for i, chunk_text in enumerate(text_chunks):
            # Find chunk position in original text
            start_pos = content.find(chunk_text, current_pos)
            end_pos = start_pos + len(chunk_text)
            current_pos = end_pos

            # Create embedded chunk
            chunk = EmbeddedChunk(
                id=chunk_id + i,
                note_id=note_id,
                title=title,
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                vector=self.embedder.embed(chunk_text),
            )
            chunks.append(chunk)

        return note, chunks

    @staticmethod
    def _build_complete_mapping(files: list[Path], folder: Path) -> dict[str, str]:
        """Build a complete mapping from note names/stems to IDs, including assets.

        Args:
            files: List of markdown files
            folder: Base folder path

        Returns:
            Dictionary mapping names/paths to IDs
        """
        mapping = {}

        # Map markdown files
        for file in files:
            note_id = md5(str(file.relative_to(folder)).encode()).hexdigest()
            mapping[file.stem] = note_id
            mapping[file.name] = note_id
            mapping[str(file.relative_to(folder))] = note_id

        # Map image files
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff"}
        for ext in image_extensions:
            for image_file in folder.rglob(f"*{ext}"):
                image_id = f"image:{image_file.relative_to(folder)}"
                mapping[image_file.stem] = image_id
                mapping[image_file.name] = image_id
                mapping[str(image_file.relative_to(folder))] = image_id

        # Map excalidraw files
        for excalidraw_file in folder.rglob("*.excalidraw"):
            excalidraw_id = f"excalidraw:{excalidraw_file.relative_to(folder)}"
            mapping[excalidraw_file.stem] = excalidraw_id
            mapping[excalidraw_file.name] = excalidraw_id
            mapping[str(excalidraw_file.relative_to(folder))] = excalidraw_id

        return mapping
