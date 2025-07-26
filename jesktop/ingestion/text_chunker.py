"""Text chunking service for markdown content."""

import re

import tiktoken


class TextChunker:
    """Service for splitting markdown text into chunks while preserving document structure."""

    def __init__(self, max_tokens: int = 1000, overlap: int = 100, model: str = "gpt-3.5-turbo"):
        """Initialize the text chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            model: Model name for tokenizer
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.enc = tiktoken.encoding_for_model(model)

    def chunk_text(self, text: str) -> list[str]:
        """Split Markdown text into chunks, trying to maintain document structure.

        Args:
            text: Input Markdown text

        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        current_tokens = 0

        # First try splitting on headers
        for section in self._split_on_headers(text):
            section_tokens = len(self.enc.encode(section))

            if section_tokens > self.max_tokens:
                # If section is too large, split on paragraphs
                for paragraph in self._split_on_paragraphs(section):
                    para_tokens = len(self.enc.encode(paragraph))

                    if para_tokens > self.max_tokens:
                        # If paragraph is too large, split on sentences
                        for sentence in self._split_on_sentences(paragraph):
                            current_chunk, current_tokens, new_chunks = self._process_text_chunk(
                                text=sentence,
                                current_chunk=current_chunk,
                                current_tokens=current_tokens,
                            )
                            chunks.extend(new_chunks)
                    else:
                        current_chunk, current_tokens, new_chunks = self._process_text_chunk(
                            text=paragraph,
                            current_chunk=current_chunk,
                            current_tokens=current_tokens,
                        )
                        chunks.extend(new_chunks)
            else:
                current_chunk, current_tokens, new_chunks = self._process_text_chunk(
                    text=section,
                    current_chunk=current_chunk,
                    current_tokens=current_tokens,
                )
                chunks.extend(new_chunks)

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        return self._add_chunk_overlap(chunks)

    @staticmethod
    def _split_on_headers(text: str) -> list[str]:
        """Split Markdown text into sections based on headers."""
        header_pattern = r"^#{1,6}\s+.+$"
        sections = re.split(f"(?={header_pattern})", text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]

    @staticmethod
    def _split_on_paragraphs(text: str) -> list[str]:
        """Split text into paragraphs while preserving list structure."""
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

    @staticmethod
    def _split_on_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _process_text_chunk(
        self, *, text: str, current_chunk: str, current_tokens: int
    ) -> tuple[str, int, list[str]]:
        """Process a text chunk and return updated state."""
        text = text.strip()
        if not text:
            return current_chunk, current_tokens, []

        text_tokens = len(self.enc.encode(text))
        chunks = []

        if current_tokens + text_tokens > self.max_tokens:
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

    def _add_chunk_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlapping context between chunks."""
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                prev_tokens = self.enc.encode(prev_chunk)[-self.overlap :]
                context = self.enc.decode(prev_tokens)
                chunk = f"Previous context: {context}\n\n{chunk}"
            overlapped_chunks.append(chunk)
        return overlapped_chunks
