"""Tests for image pattern matching functionality."""

import pytest

from scripts.ingest import extract_image_paths


@pytest.mark.parametrize(
    "description,content,expected_paths",
    [
        (
            "Simple markdown image",
            "![alt text](image.png)",
            ["image.png"],
        ),
        (
            "Image with spaces",
            "![](path with spaces/image.jpg)",
            ["path with spaces/image.jpg"],
        ),
        (
            "Image with parentheses",
            "![](path/Image (2).png)",
            ["path/Image (2).png"],
        ),
        (
            "Full path with attachments",
            "![](Z - Attachements/HPC sim mesh.assets/Image (2).png)",
            ["Z - Attachements/HPC sim mesh.assets/Image (2).png"],
        ),
        (
            "HTML image",
            '<img src="image.png" alt="alt text">',
            ["image.png"],
        ),
        (
            "HTML image with single quotes",
            "<img src='image.jpg' alt='alt text'>",
            ["image.jpg"],
        ),
        (
            "Multiple images",
            "Here's an image ![](image1.png) and another ![](path/Image (2).png)",
            ["image1.png", "path/Image (2).png"],
        ),
        (
            "Mixed markdown and HTML",
            '![](image1.png) and <img src="image2.jpg">',
            ["image1.png", "image2.jpg"],
        ),
        (
            "No images",
            "Just some text without images",
            [],
        ),
        (
            "External URLs",
            "![](https://example.com/image.jpg)",
            [],  # External URLs should be filtered out
        ),
    ],
)
def test_extract_image_paths(description, content, expected_paths):  # noqa
    paths = extract_image_paths(content)
    assert paths == expected_paths, f"Failed test case: {description}"
