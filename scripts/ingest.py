"""CLI for getting embeddings for sample data and saving to either local Vector DB or a DBX Mosaic Vector DB"""

import argparse
import logging
import mimetypes
import re
from hashlib import md5, sha256
from pathlib import Path
from urllib.parse import unquote

import tiktoken

from jesktop.config import settings
from jesktop.embedders.base import Embedder
from jesktop.embedders.voyage_embedder import VoyageEmbedder
from jesktop.image_store.local import LocalImageStore
from jesktop.vector_dbs.local_db import LocalVectorDB
from jesktop.vector_dbs.schemas import (
    EmbeddedChunk,
    Image,
    Note,
    NoteRelationship,
    RelationshipGraph,
)

logger = logging.getLogger(__name__)


def split_on_headers(text: str) -> list[str]:
    """Split markdown text into sections based on headers."""
    header_pattern = r"^#{1,6}\s+.+$"
    sections = re.split(f"(?={header_pattern})", text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def split_on_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs while preserving list structure."""
    # Split on double newlines that aren't part of a list
    parts = []
    current_part = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        is_empty = not line.strip()
        next_is_list = i < len(lines) - 1 and bool(re.match(r"^[\s]*[-*+]|\d+\.", lines[i + 1]))

        current_part.append(line)

        # Split if we have an empty line followed by a non-list item
        if is_empty and i < len(lines) - 1 and not next_is_list:
            if current_part:
                parts.append("\n".join(current_part))
                current_part = []

    if current_part:
        parts.append("\n".join(current_part))

    return [p.strip() for p in parts if p.strip()]


def split_on_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def add_chunk_overlap(chunks: list[str], enc: tiktoken.Encoding, overlap: int) -> list[str]:
    """Add overlapping context between chunks."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_chunk = chunks[i - 1]
            prev_tokens = enc.encode(prev_chunk)[-overlap:]
            context = enc.decode(prev_tokens)
            chunk = f"Previous context: {context}\n\n{chunk}"
        overlapped_chunks.append(chunk)
    return overlapped_chunks


def process_text_chunk(
    *, text: str, current_chunk: str, current_tokens: int, max_tokens: int, enc: tiktoken.Encoding
) -> tuple[str, int, list[str]]:
    """Process a text chunk and return updated state."""
    text = text.strip()
    if not text:
        return current_chunk, current_tokens, []

    text_tokens = len(enc.encode(text))
    chunks = []

    if current_tokens + text_tokens > max_tokens:
        if current_chunk:
            chunks.append(current_chunk.strip())
        current_chunk = text
        current_tokens = text_tokens
    else:
        if current_chunk:
            current_chunk += "\n\n"
        current_chunk += text
        current_tokens += text_tokens

    return current_chunk, current_tokens, chunks


def split_markdown_into_chunks(text: str, max_tokens: int = 1000, overlap: int = 100) -> list[str]:
    """Split markdown text into chunks, trying to maintain document structure."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    chunks = []
    current_chunk = ""
    current_tokens = 0

    # First try splitting on headers
    for section in split_on_headers(text):
        section_tokens = len(enc.encode(section))

        if section_tokens > max_tokens:
            # If section is too large, split on paragraphs
            for paragraph in split_on_paragraphs(section):
                para_tokens = len(enc.encode(paragraph))

                if para_tokens > max_tokens:
                    # If paragraph is too large, split on sentences
                    for sentence in split_on_sentences(paragraph):
                        current_chunk, current_tokens, new_chunks = process_text_chunk(
                            text=sentence,
                            current_chunk=current_chunk,
                            current_tokens=current_tokens,
                            max_tokens=max_tokens,
                            enc=enc,
                        )
                        chunks.extend(new_chunks)
                else:
                    current_chunk, current_tokens, new_chunks = process_text_chunk(
                        text=paragraph,
                        current_chunk=current_chunk,
                        current_tokens=current_tokens,
                        max_tokens=max_tokens,
                        enc=enc,
                    )
                    chunks.extend(new_chunks)
        else:
            current_chunk, current_tokens, new_chunks = process_text_chunk(
                text=section,
                current_chunk=current_chunk,
                current_tokens=current_tokens,
                max_tokens=max_tokens,
                enc=enc,
            )
            chunks.extend(new_chunks)

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    return add_chunk_overlap(chunks, enc, overlap)


def extract_image_paths(content: str) -> list[str]:
    """Extract image paths from markdown, HTML, and wikilink content.

    Args:
        content: String containing markdown or HTML content with image references.

    Returns:
        List of image paths found in the content.
    """
    # Pattern matches: ![alt](path), <img src="path">, and ![[image.ext]]
    image_pattern = (
        r"!\[([^\]]*)\]\(([^\(\)]*(?:\([^\(\)]*\)[^\(\)]*)*)\)|"  # ![alt](path)
        r'<img[^>]+src=[\'"](.*?)[\'"][^>]*>|'  # <img src="path">
        r"\!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff))\]\]"  # ![[image.ext]]
    )
    paths = []

    for match in re.finditer(image_pattern, content):
        if match.group(2):  # Markdown syntax ![alt](path)
            img_path = match.group(2)
        elif match.group(3):  # HTML syntax <img src="path">
            img_path = match.group(3)
        elif match.group(4):  # Wikilink syntax ![[image.ext]]
            img_path = match.group(4)
        else:
            continue

        img_path = img_path.strip()

        if not img_path.startswith(("http://", "https://")):
            paths.append(img_path)

    return paths


def extract_wikilinks(content: str) -> list[str]:
    """Extract Obsidian wikilink references from markdown content.

    Extracts links in the form of [[link name]] or [[link name|display text]].
    """
    wikilink_pattern = r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]"
    return re.findall(wikilink_pattern, content)


def extract_embedded_content(content: str) -> list[str]:
    """Extract embedded content references from markdown content.

    Extracts embeds in the form of ![[content name]].
    """
    embed_pattern = r"\!\[\[([^\]]+)\]\]"
    return re.findall(embed_pattern, content)


def extract_excalidraw_refs(content: str) -> list[str]:
    """Extract Obsidian excalidraw references from markdown content.

    The references are in the form of ![[(path/to/file.excalidraw]].
    """
    excalidraw_pattern = r"\!\[\[([^\]]+\.excalidraw)\]\]"
    return re.findall(excalidraw_pattern, content)


def resolve_note_references(links: list[str], note_mapping: dict[str, str]) -> list[str]:
    """Convert note names/paths to note IDs using a mapping dictionary.

    Args:
        links: List of note names or paths from wikilinks
        note_mapping: Dictionary mapping note names/stems to note IDs/asset references

    Returns:
        List of resolved note IDs and asset references
    """
    resolved_ids = []
    for link in links:
        # Try exact match first
        if link in note_mapping:
            resolved_ids.append(note_mapping[link])
        else:
            # Try with .md extension
            md_link = f"{link}.md"
            if md_link in note_mapping:
                resolved_ids.append(note_mapping[md_link])
            else:
                # Try as filename stem
                for path, note_id in note_mapping.items():
                    if Path(path).stem == link:
                        resolved_ids.append(note_id)
                        break
                else:
                    # Check if it might be an image or excalidraw file with different casing
                    # or in a different location - be more lenient for assets
                    found = False
                    link_lower = link.lower()
                    for path, asset_id in note_mapping.items():
                        if asset_id.startswith(("image:", "excalidraw:")) and (
                            path.lower() == link_lower or Path(path).stem.lower() == link_lower
                        ):
                            resolved_ids.append(asset_id)
                            found = True
                            break

                    if not found:
                        logger.warning(f"Could not resolve wikilink: {link}")

    return resolved_ids


def calculate_relationship_strength(source_content: str, target_name: str) -> float:
    """Calculate relationship strength based on frequency and context.

    Args:
        source_content: Full content of the source note
        target_name: Name of the target note

    Returns:
        Relationship strength between 0.0 and 1.0
    """
    # Count occurrences of the target in the source
    occurrences = len(re.findall(re.escape(target_name), source_content, re.IGNORECASE))

    # Base strength on frequency, capped at 1.0
    base_strength = min(occurrences * 0.3, 1.0)

    # Boost if mentioned in headers
    header_mentions = len(
        re.findall(f"#{1, 6}.*{re.escape(target_name)}", source_content, re.IGNORECASE)
    )
    header_boost = header_mentions * 0.2

    return min(base_strength + header_boost, 1.0)


def extract_relationship_context(content: str, target_name: str, context_chars: int = 100) -> str:
    """Extract surrounding context for a relationship mention.

    Args:
        content: Full content of the note
        target_name: Name of the target being referenced
        context_chars: Number of characters before/after to include

    Returns:
        Context string around the first mention
    """
    # Find first mention of target
    match = re.search(re.escape(target_name), content, re.IGNORECASE)
    if not match:
        return ""

    start = max(0, match.start() - context_chars)
    end = min(len(content), match.end() + context_chars)

    context = content[start:end].strip()

    # Clean up context - remove newlines, extra spaces
    context = re.sub(r"\s+", " ", context)

    return context


def process_excalidraw_refs_in_note(
    *,
    content: str,
    note_id: str,
    file: Path,  # noqa: ARG001
    image_store: LocalImageStore,
) -> None:
    """Save corresponding png image for each excalidraw reference in the note."""
    for img_path in extract_excalidraw_refs(content):
        img_path = Path(f"{unquote(img_path)}.png")

        # A png image is automatically generated for each excalidraw file
        excalidraw_image_path = Path("data/notes/Z - Attachements") / img_path
        if not excalidraw_image_path.exists():
            logger.warning(f"Image not found at normal path: {excalidraw_image_path}")
            continue
        # Read image content and calculate hash
        with open(excalidraw_image_path, "rb") as f:
            image_content = f.read()
            image_hash = sha256(image_content).hexdigest()

        # Determine mime type
        mime_type, _ = mimetypes.guess_type(str(excalidraw_image_path))
        if not mime_type or not mime_type.startswith("image/"):
            logger.warning(f"Not an image or unknown type: {excalidraw_image_path}")
            continue

        # Store image in database
        image = Image(
            id=image_hash,
            note_id=note_id,
            content=image_content,
            mime_type=mime_type,
            relative_path=str(img_path),
            absolute_path=str(excalidraw_image_path),
        )
        image_store.save_image(image)
        logger.info(f"Stored image {img_path} with hash {image_hash}")


def process_images_in_note(
    *, content: str, note_id: str, file: Path, image_store: LocalImageStore
) -> None:
    """Store images from markdown in the database without modifying content."""
    for img_path in extract_image_paths(content):
        img_path = Path(unquote(img_path))

        excalidraw_asset_dir_path = f"{file.name.split('.')[0]}.assets"

        abs_asset_img_path = (
            Path("data/notes/Z - Attachements") / excalidraw_asset_dir_path / img_path
        )
        abs_asset_img_path2 = Path("data/notes/") / img_path
        abs_direct_attachment_path = Path("data/notes/Z - Attachements") / img_path
        abs_normal_img_path = (file.parent / img_path).resolve()

        if abs_asset_img_path2.exists():
            abs_img_path = abs_asset_img_path2
        elif abs_direct_attachment_path.exists():
            abs_img_path = abs_direct_attachment_path
        elif abs_asset_img_path.exists():
            abs_img_path = abs_asset_img_path
        elif abs_normal_img_path.exists():
            abs_img_path = abs_normal_img_path
        else:
            logger.warning(
                f"Image not found at paths: {abs_normal_img_path}, {abs_asset_img_path}, {abs_direct_attachment_path}, {abs_asset_img_path2}"
            )
            continue

        # Read image content and calculate hash
        with open(abs_img_path, "rb") as f:
            image_content = f.read()
            image_hash = sha256(image_content).hexdigest()

        # Determine mime type
        mime_type, _ = mimetypes.guess_type(str(abs_img_path))
        if not mime_type or not mime_type.startswith("image/"):
            logger.warning(f"Not an image or unknown type: {abs_img_path}")
            continue

        # Store image in database
        image = Image(
            id=image_hash,
            note_id=note_id,
            content=image_content,
            mime_type=mime_type,
            relative_path=str(img_path),  # Store the original relative path
            absolute_path=str(abs_img_path),
        )
        image_store.save_image(image)
        logger.info(f"Stored image {img_path} with hash {image_hash}")


def replace_image_paths(content: str, note_id: str) -> str:
    """Replace all image paths in markdown with API endpoints."""

    def replace_match(match: re.Match[str]) -> str:
        alt_text = match.group(1) or ""
        # Check all possible groups for the image path
        img_path = match.group(2) or match.group(3) or match.group(4) or match.group(5) or ""
        img_path = unquote(img_path.strip())

        # Skip external URLs
        if img_path.startswith(("http://", "https://")):
            return match.group(0)

        # Handle excalidraw files - convert to PNG
        if img_path.endswith(".excalidraw"):
            img_path = img_path + ".png"

        # Clean up the path and ensure correct encoding
        img_path = str(Path(img_path))
        api_path = f"/api/images/{note_id}/{img_path}"
        return f"![{alt_text}]({api_path})"

    # Replace both markdown, HTML, and wikilink image syntax
    pattern = (
        r"!\[([^\]]*)\]\(([^\(\)]*(?:\([^\(\)]*\)[^\(\)]*)*)\)|"  # ![alt](path)
        r'<img[^>]+src=[\'"](.*?)[\'"][^>]*>|'  # <img src="path">
        r"\!\[\[([^\]]+\.excalidraw)\]\]|"  # ![[file.excalidraw]]
        r"\!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff))\]\]"  # ![[image.ext]]
    )
    return re.sub(pattern, replace_match, content)


# def replace_excalidraw_refs_in_note(content: str, note_id: str) -> str:
#     """Replace excalidraw references with API endpoints."""
#
#     def replace_match(match):
#         img_path = match.group(1)  # Get the captured group without brackets
#         img_path = Path(f"{unquote(img_path)}.png")
#         api_path = f"/api/images/{note_id}/{img_path}"
#         return f"![]({api_path})"
#
#     # Replace excalidraw references of the form ![[file.excalidraw]]
#     pattern = r"\!\[\[([^\]]+\.excalidraw)\]\]"
#     return re.sub(pattern, replace_match, content)


def process_file(
    *,
    file: Path,
    folder: Path,
    embedder: Embedder,
    chunk_id: int,
    image_store: LocalImageStore,
    note_mapping: dict[str, str] | None = None,
) -> tuple[Note, list[EmbeddedChunk]]:
    """Process a single markdown file."""
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

    process_images_in_note(content=content, note_id=note_id, file=file, image_store=image_store)
    process_excalidraw_refs_in_note(
        content=content, note_id=note_id, file=file, image_store=image_store
    )

    # Extract relationships if mapping is available
    outbound_links = []
    embedded_content = []
    if note_mapping:
        # Extract wikilinks and resolve to note IDs
        wikilinks = extract_wikilinks(content)
        outbound_links = resolve_note_references(wikilinks, note_mapping)

        # Extract embedded content (images, excalidraw)
        embeds = extract_embedded_content(content)
        embedded_content = [sha256(embed.encode()).hexdigest() for embed in embeds]

    content = replace_image_paths(content, note_id)
    # content = replace_excalidraw_refs_in_note(content=content, note_id=note_id)

    # Extract folder path for hierarchical relationships
    relative_path = file.relative_to(folder)
    folder_path = str(relative_path.parent) if relative_path.parent != Path(".") else ""

    # Create Note object
    note = Note(
        id=note_id,
        title=title,
        path=str(file),
        content=content,
        metadata={
            "created": file.stat().st_ctime,
            "modified": file.stat().st_mtime,
        },
        outbound_links=outbound_links,
        embedded_content=embedded_content,
        folder_path=folder_path,
    )

    # Split the text into chunks
    chunks = []
    text_chunks = split_markdown_into_chunks(content)
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
            vector=embedder.embed(chunk_text),
        )
        chunks.append(chunk)

    return note, chunks


def build_note_mapping(files: list[Path], folder: Path) -> dict[str, str]:
    """Build a mapping from note names/stems to note IDs for reference resolution.

    Also includes image and excalidraw files that can be referenced via wikilinks.
    """
    mapping = {}

    for file in files:
        # Generate note ID (same logic as in process_file)
        note_id = md5(str(file.relative_to(folder)).encode()).hexdigest()

        # Add mappings for different ways the note might be referenced
        mapping[file.stem] = note_id  # filename without extension
        mapping[file.name] = note_id  # filename with extension
        mapping[str(file.relative_to(folder))] = note_id  # relative path

    # Add image files to mapping - they can be referenced via wikilinks
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff"}
    for ext in image_extensions:
        image_files = list(folder.rglob(f"*{ext}"))
        for image_file in image_files:
            # Use a special prefix to distinguish image files from notes
            image_id = f"image:{image_file.relative_to(folder)}"
            mapping[image_file.stem] = image_id
            mapping[image_file.name] = image_id
            mapping[str(image_file.relative_to(folder))] = image_id

    # Add excalidraw files to mapping
    excalidraw_files = list(folder.rglob("*.excalidraw"))
    for excalidraw_file in excalidraw_files:
        # Use a special prefix to distinguish excalidraw files from notes
        excalidraw_id = f"excalidraw:{excalidraw_file.relative_to(folder)}"
        mapping[excalidraw_file.stem] = excalidraw_id
        mapping[excalidraw_file.name] = excalidraw_id
        mapping[str(excalidraw_file.relative_to(folder))] = excalidraw_id

    return mapping


def build_relationships(notes: dict[str, Note]) -> RelationshipGraph:
    """Build relationship graph from processed notes."""
    relationships = []
    note_clusters = {}

    # Build note clusters by folder
    for note in notes.values():
        if note.folder_path:
            if note.folder_path not in note_clusters:
                note_clusters[note.folder_path] = []
            note_clusters[note.folder_path].append(note.id)

    # Build relationships from outbound links
    for note in notes.values():
        for target_id in note.outbound_links:
            # Only create note-to-note relationships, skip asset references
            if target_id in notes and not target_id.startswith(("image:", "excalidraw:")):
                target_note = notes[target_id]

                # Calculate relationship strength and context
                strength = calculate_relationship_strength(note.content, target_note.title)
                context = extract_relationship_context(note.content, target_note.title)

                relationship = NoteRelationship(
                    source_note_id=note.id,
                    target_note_id=target_id,
                    relationship_type="wikilink",
                    context=context,
                    strength=strength,
                )
                relationships.append(relationship)

    return RelationshipGraph(
        relationships=relationships,
        note_clusters=note_clusters,
    )


def update_inbound_links(notes: dict[str, Note], relationships: list[NoteRelationship]) -> None:
    """Update inbound_links for all notes based on relationships."""
    # Clear existing inbound links
    for note in notes.values():
        note.inbound_links = []

    # Build inbound links from relationships
    for rel in relationships:
        if rel.target_note_id in notes:
            notes[rel.target_note_id].inbound_links.append(rel.source_note_id)


def process_folder(
    folder: Path, embedder: Embedder
) -> tuple[dict[str, Note], dict[int, EmbeddedChunk], LocalImageStore, RelationshipGraph]:
    """Process all markdown files in a folder."""
    files = list(folder.rglob("*.md"))

    # Exclude excalidraw files
    files = [f for f in files if not f.name.endswith(".excalidraw.md")]

    # Build note mapping for relationship resolution
    note_mapping = build_note_mapping(files, folder)

    notes = {}
    chunks = {}
    chunk_id = 0
    image_store = LocalImageStore()

    # First pass: process files with relationship extraction
    for file in files:
        note, note_chunks = process_file(
            file=file,
            folder=folder,
            embedder=embedder,
            chunk_id=chunk_id,
            image_store=image_store,
            note_mapping=note_mapping,
        )
        notes[note.id] = note
        for chunk in note_chunks:
            chunks[chunk.id] = chunk
        chunk_id += len(note_chunks)

    # Second pass: build relationship graph and update inbound links
    relationship_graph = build_relationships(notes)
    update_inbound_links(notes, relationship_graph.relationships)

    return notes, chunks, image_store, relationship_graph


def main(
    in_folder: str,
    local_outfile_vector_db: str,
    local_outfile_image_store: str,
) -> None:
    """Main function."""
    folder = Path(in_folder)
    vector_db_output = Path(local_outfile_vector_db)
    image_store_output = Path(local_outfile_image_store)

    # Create embedder
    embedder = VoyageEmbedder(api_key=settings.voyage_ai_api_key)

    # Process files
    notes, chunks, image_store, relationship_graph = process_folder(folder, embedder)

    # Create and save vector database and image store
    db = LocalVectorDB(notes=notes, embedded_chunks=chunks, relationship_graph=relationship_graph)
    db.save(str(vector_db_output))
    image_store.save(str(image_store_output))

    print(f"Processed {len(notes)} notes into {len(chunks)} chunks")
    print(f"Found {len(relationship_graph.relationships)} relationships")
    print(f"Created {len(relationship_graph.note_clusters)} folder clusters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-folder", type=str, required=True, help="Folder containing markdown files"
    )
    parser.add_argument(
        "--outfile-vector-db",
        type=str,
        required=False,
        help="Local output vector db file",
        default=settings.local_vector_db_path,
    )
    parser.add_argument(
        "--outfile-image-store",
        type=str,
        required=False,
        help="Local output image store file",
        default=settings.local_image_store_path,
    )

    args = parser.parse_args()

    main(
        in_folder=args.in_folder,
        local_outfile_vector_db=args.outfile_vector_db,
        local_outfile_image_store=args.outfile_image_store,
    )
