"""Tests for the various PDF parsers."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pytest
from langchain_community.document_loaders.parsers import BaseImageBlobParser

from langchain_unstructured.pdf import UnstructuredPDFLoader

if TYPE_CHECKING:
    from PIL.Image import Image

# PDFs to test parsers on.
EXAMPLE_DOCS_DIRECTORY = Path(__file__).parent.parent / "examples"

HELLO_PDF = EXAMPLE_DOCS_DIRECTORY / "hello.pdf"

LAYOUT_PARSER_PAPER_PDF = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"
TITLE_LEVEL_PDF = EXAMPLE_DOCS_DIRECTORY / "title_level.pdf"

LAYOUT_PARSER_PAPER_PASSWORD_PDF = (
    EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper-password.pdf"
)

DUPLICATE_CHARS = EXAMPLE_DOCS_DIRECTORY / "duplicate-chars.pdf"

UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")


class EmptyImageBlobParser(BaseImageBlobParser):
    def _analyze_image(self, img: "Image") -> str:
        return "Hello world"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "auto"},
        {"strategy": "fast"},
        {"strategy": "hi_res"},
        {"strategy": "ocr_only"},
    ],
)
@pytest.mark.local
def test_unstructured_pdf_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    file_path = LAYOUT_PARSER_PAPER_PDF
    loader = UnstructuredPDFLoader(file_path, mode="single", **kwargs)
    docs = loader.load()

    assert len(docs) == 1


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
def test_unstructured_pdf_loader_via_api() -> None:
    file_path = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"
    loader = UnstructuredPDFLoader(
        file_path=file_path,
        api_key=UNSTRUCTURED_API_KEY,
        partition_via_api=True,
        mode="elements",
        # Unstructured kwargs
        strategy="fast",
    )

    docs = loader.load()

    assert len(docs) > 1
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"


@pytest.fixture()
def get_post_processor() -> Callable[[str], str]:
    def append_the_end(text: str) -> str:
        return text + "THE END!"

    return append_the_end
