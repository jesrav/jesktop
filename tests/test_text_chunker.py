from jesktop.ingestion.text_chunker import TextChunker


def test_text_chunker_basic_functionality() -> None:
    """Test basic text chunking functionality."""
    chunker = TextChunker(max_tokens=20, overlap=5)

    text = """# Header 1

This is a paragraph under header 1 with lots of content that should definitely exceed the token limit.

## Header 2

This is another paragraph under header 2 with even more content to ensure we get multiple chunks.

### Header 3

And this is a third paragraph with additional content to make sure we have enough text to split."""

    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 1
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_text_chunker_small_text() -> None:
    """Test chunking with text smaller than max tokens."""
    chunker = TextChunker(max_tokens=1000, overlap=100)

    text = "This is a small text that should fit in one chunk."

    chunks = chunker.chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].strip() == text.strip()


def test_text_chunker_header_splitting() -> None:
    """Test that chunker respects header boundaries."""
    chunker = TextChunker(max_tokens=10, overlap=2)

    text = """# First Header

This is a long paragraph under the first header with lots of content that should definitely exceed our very small token limit and cause multiple chunks to be created.

# Second Header

This is another long paragraph under the second header with even more content to ensure we get proper splitting behavior and multiple chunks with header boundaries respected."""

    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 2

    first_header_chunks = [chunk for chunk in chunks if "# First Header" in chunk]
    second_header_chunks = [chunk for chunk in chunks if "# Second Header" in chunk]

    assert len(first_header_chunks) >= 1
    assert len(second_header_chunks) >= 1

    header_starting_chunks = [chunk for chunk in chunks if chunk.strip().startswith("#")]
    assert len(header_starting_chunks) >= 1


def test_text_chunker_custom_parameters() -> None:
    """Test TextChunker with different parameters produces different results."""
    small_chunker = TextChunker(max_tokens=10, overlap=2)
    large_chunker = TextChunker(max_tokens=100, overlap=2)

    text = """This is a long text that contains multiple sentences and paragraphs.
    
It should definitely exceed a 10-token limit but stay within a 100-token limit.
The small chunker should create multiple chunks while the large chunker creates fewer."""

    small_chunks = small_chunker.chunk_text(text)
    large_chunks = large_chunker.chunk_text(text)

    assert len(small_chunks) > len(large_chunks)

    assert len(small_chunks) >= 1
    assert len(large_chunks) >= 1

    small_content = " ".join(small_chunks)
    large_content = " ".join(large_chunks)

    key_words = ["long text", "sentences", "paragraphs", "token limit"]
    for word in key_words:
        assert word in small_content
        assert word in large_content


def test_text_chunker_empty_text() -> None:
    """Test chunking empty or whitespace-only text."""
    chunker = TextChunker()

    chunks = chunker.chunk_text("")
    assert chunks == []

    chunks = chunker.chunk_text("   \n\n   ")
    assert chunks == []


def test_text_chunker_overlap_functionality() -> None:
    """Test that overlap functionality works correctly."""
    with_overlap = TextChunker(max_tokens=15, overlap=5)
    without_overlap = TextChunker(max_tokens=15, overlap=0)

    text = """This is the first section with some content that will span multiple chunks.

This is the second section with more content that will also span multiple chunks.

This is the third section with even more content to ensure proper chunking behavior."""

    chunks_with_overlap = with_overlap.chunk_text(text)
    chunks_without_overlap = without_overlap.chunk_text(text)

    if len(chunks_with_overlap) > 1:
        overlapped_chunks = [
            chunk for chunk in chunks_with_overlap[1:] if "Previous context:" in chunk
        ]
        assert len(overlapped_chunks) > 0

    if len(chunks_without_overlap) > 1:
        overlapped_chunks = [
            chunk for chunk in chunks_without_overlap if "Previous context:" in chunk
        ]
        assert len(overlapped_chunks) == 0
