"""Content extraction service for markdown content."""

import logging
import mimetypes
import re
from hashlib import sha256
from pathlib import Path
from typing import List
from urllib.parse import unquote

from jesktop.domain.image import Image
from jesktop.image_store.base import ImageStore

from .path_resolver import PathResolver

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Service for extracting various content types from markdown text."""

    @staticmethod
    def extract_image_paths(content: str) -> List[str]:
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

    @staticmethod
    def extract_wikilinks(content: str) -> List[str]:
        """Extract Obsidian wikilink references from markdown content.

        Extracts links in the form of [[link name]] or [[link name|display text]].

        Args:
            content: Markdown content to extract wikilinks from

        Returns:
            List of wikilink targets
        """
        wikilink_pattern = r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]"
        return re.findall(wikilink_pattern, content)

    @staticmethod
    def extract_embedded_content(content: str) -> List[str]:
        """Extract embedded content references from markdown content.

        Extracts embeds in the form of ![[content name]].

        Args:
            content: Markdown content to extract embeds from

        Returns:
            List of embedded content references
        """
        embed_pattern = r"\!\[\[([^\]]+)\]\]"
        return re.findall(embed_pattern, content)

    @staticmethod
    def extract_excalidraw_refs(content: str) -> List[str]:
        """Extract Obsidian excalidraw references from markdown content.

        The references are in the form of ![[(path/to/file.excalidraw]].

        Args:
            content: Markdown content to extract excalidraw references from

        Returns:
            List of excalidraw file references
        """
        excalidraw_pattern = r"\!\[\[([^\]]+\.excalidraw)\]\]"
        return re.findall(excalidraw_pattern, content)

    @staticmethod
    def replace_image_paths(content: str, note_id: str) -> str:
        """Replace all image paths in markdown with API endpoints.

        Args:
            content: Markdown content containing image references
            note_id: ID of the note containing the images

        Returns:
            Content with image paths replaced by API endpoints
        """

        def replace_match(match: re.Match[str]) -> str:
            alt_text = match.group(1) or ""
            # Check all possible groups for the image path
            img_path = match.group(2) or match.group(3) or match.group(4) or match.group(5) or ""
            img_path = img_path.strip()

            # Skip external URLs
            if img_path.startswith(("http://", "https://")):
                return match.group(0)

            # Handle excalidraw files - convert to PNG
            if img_path.endswith(".excalidraw"):
                img_path = img_path + ".png"

            # Clean up the path and ensure correct encoding
            from pathlib import Path
            from urllib.parse import unquote

            img_path = str(Path(unquote(img_path)))
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

    @staticmethod
    def _process_single_image(
        *,
        image_path: Path,
        original_path: str,
        note_id: str,
        image_store: ImageStore,
    ) -> None:
        """Process a single image file and store it in the image store.

        Common logic for processing both regular images and excalidraw images.
        """
        # Read image content and calculate hash
        with open(image_path, "rb") as f:
            image_content = f.read()
            image_hash = sha256(image_content).hexdigest()

        # Determine mime type
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            logger.warning(f"Not an image or unknown type: {image_path}")
            return

        # Store image in database
        image = Image(
            id=image_hash,
            note_id=note_id,
            content=image_content,
            mime_type=mime_type,
            relative_path=original_path,
            absolute_path=str(image_path),
        )
        image_store.add_image(image)
        logger.info(f"Stored image {original_path} with hash {image_hash}")

    def process_excalidraw_refs_in_note(
        self,
        *,
        content: str,
        note_id: str,
        file: Path,
        image_store: ImageStore,
        path_resolver: PathResolver,
    ) -> None:
        """Save corresponding png image for each excalidraw reference in the note."""
        for excalidraw_ref in self.extract_excalidraw_refs(content):
            # Convert excalidraw reference to corresponding PNG path
            png_path = f"{unquote(excalidraw_ref)}.png"

            # Use PathResolver to find the PNG file
            resolved_path = path_resolver.resolve_image_path(file, png_path)

            if resolved_path is None:
                logger.warning(f"Excalidraw PNG not found: {png_path}")
                continue

            self._process_single_image(
                image_path=resolved_path,
                original_path=png_path,
                note_id=note_id,
                image_store=image_store,
            )

    def process_images_in_note(
        self,
        *,
        content: str,
        note_id: str,
        file: Path,
        image_store: ImageStore,
        path_resolver: PathResolver,
    ) -> None:
        """Store images from markdown in the database without modifying content."""
        for img_path in self.extract_image_paths(content):
            # Use PathResolver to resolve image path
            resolved_path = path_resolver.resolve_image_path(file, img_path)

            if resolved_path is None:
                logger.warning(f"Image not found: {img_path}")
                continue

            self._process_single_image(
                image_path=resolved_path,
                original_path=str(img_path),
                note_id=note_id,
                image_store=image_store,
            )
