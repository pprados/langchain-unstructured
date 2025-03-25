"""Tests for the various PDF parsers."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Type

import pytest
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import BaseImageBlobParser

from langchain_unstructured.parsers import UnstructuredPDFParser
from langchain_unstructured.parsers.base import (
    Type_image,
    _UnstructuredDocumentParser,
    satisfies_min_pdfminer_six_version, Type_extract_tables,
)

if TYPE_CHECKING:
    from PIL.Image import Image

# PDFs to test parsers on.
EXAMPLE_DOCS_DIRECTORY = Path(__file__).parent.parent.parent / "examples"

HELLO_PDF = EXAMPLE_DOCS_DIRECTORY / "hello.pdf"

LAYOUT_PARSER_PAPER_PDF = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"

LAYOUT_PARSER_PAPER_PASSWORD_PDF = (
    EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper-password.pdf"
)

DUPLICATE_CHARS = EXAMPLE_DOCS_DIRECTORY / "duplicate-chars.pdf"


class EmptyImageBlobParser(BaseImageBlobParser):
    def _analyze_image(self, img: "Image") -> str:
        return "Hello world"


def _assert_with_parser(parser: BaseBlobParser, splits_by_page: bool = True) -> None:
    """Standard tests to verify that the given parser works.

    Args:
        parser (BaseBlobParser): The parser to test.
        splits_by_page (bool): Whether the parser splits by page or not by default.
    """
    blob = Blob.from_path(HELLO_PDF)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)
    assert parser.mode == "elements" or len(docs) == 1  # type: ignore[attr-defined]
    page_content = docs[0].page_content
    assert isinstance(page_content, str)
    # The different parsers return different amount of whitespace, so using
    # startswith instead of equals.
    assert re.findall(r"Hello\s+world!", docs[0].page_content)

    blob = Blob.from_path(LAYOUT_PARSER_PAPER_PDF)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)

    if parser.mode != "elements":  # type: ignore[attr-defined]
        if splits_by_page:
            assert len(docs) == 16
        else:
            assert len(docs) == 1
        assert "LayoutParser" in docs[0].page_content
    metadata = docs[0].metadata

    assert metadata["source"] == str(LAYOUT_PARSER_PAPER_PDF)

    if splits_by_page:
        assert metadata["page"] == 1


def _assert_with_duplicate_parser(parser: BaseBlobParser, dedupe: bool = False) -> None:
    """PDFPlumber tests to verify that duplicate characters appear or not
    Args:
        parser (BaseBlobParser): The parser to test.
        splits_by_page (bool): Whether the parser splits by page or not by default.
        dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
    """
    blob = Blob.from_path(DUPLICATE_CHARS)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)

    if dedupe:
        # use dedupe avoid duplicate characters.
        assert "1000 Series" == docs[0].page_content.split("\n")[0]
    else:
        # duplicate characters will appear in doc if not dedupe
        assert "11000000  SSeerriieess" == docs[0].page_content.split("\n")[0]


@pytest.mark.parametrize(
    "kwargs,doc_lens",
    [
        ({"mode": "single"}, 1),
        ({"mode": "page"}, 1),
        ({"mode": "elements"}, 1),
        ({"strategy": "auto"}, 1),
        ({"strategy": "fast"}, 1),
        ({"strategy": "hi_res"}, 1),
        ({"strategy": "ocr_only"}, 1),
    ],
)
@pytest.mark.skipif(
    not satisfies_min_pdfminer_six_version("20250327"),
    reason="Some PDF file crash. See pr 1081 in pdfminer.six",
)
@pytest.mark.local
def test_mode_and_extract_images_variations(
    kwargs: dict,
    doc_lens: int,
) -> None:
    image_parser = EmptyImageBlobParser()
    _test_matrix(
        UnstructuredPDFParser,
        kwargs,
        image_parser,
        images_inner_format="markdown-img",
    )


@pytest.mark.parametrize(
    "parser,kwargs",
    [
        (
            UnstructuredPDFParser,
            {"strategy": "auto"},
        ),
        (
            UnstructuredPDFParser,
            {"strategy": "fast"},
        ),
        (
            UnstructuredPDFParser,
            {"strategy": "hi_res"},
        ),
        (
            UnstructuredPDFParser,
            {"strategy": "ocr_only"},
        ),
    ],
)
@pytest.mark.local
def test_mode_and_image_formats_variations(
    parser: Type,
    kwargs: dict,
) -> None:
    images_inner_format: Type_image = "markdown-img"
    image_parser = EmptyImageBlobParser()

    _test_matrix(
        parser,
        kwargs=kwargs,
        image_parser=image_parser,
        images_inner_format=images_inner_format,
    )


def _test_matrix(
    parser_class: Type,
    kwargs: dict,
    image_parser: BaseImageBlobParser,
    images_inner_format: Type_image,
) -> None:
    """Apply the same test for all *standard* PDF parsers.

    - Try with mode `single` and `page`
    - Try with image_parser `None` or others
    """

    def _std_assert_with_parser(parser: _UnstructuredDocumentParser) -> None:
        """Standard tests to verify that the given parser works.

        Args:
            parser (BaseBlobParser): The parser to test.
        """
        blob = Blob.from_path(LAYOUT_PARSER_PAPER_PDF)
        doc_generator = parser.lazy_parse(blob)
        docs = list(doc_generator)
        metadata = docs[0].metadata
        assert metadata["source"] == str(LAYOUT_PARSER_PAPER_PDF)
        assert "creationdate" in metadata
        assert "creator" in metadata
        assert "producer" in metadata
        assert "total_pages" in metadata
        if len(docs) > 1:
            assert parser.mode == "elements" or metadata["page"] == 1
        if hasattr(parser, "extract_images") and parser.extract_images:
            images = []
            for doc in docs:
                _HTML_image = (
                    r"<img\s+[^>]*"
                    r'src="([^"]+)"(?:\s+alt="([^"]*)")?(?:\s+'
                    r'title="([^"]*)")?[^>]*>'
                )
                _markdown_image = r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"([^\"]+)\")?\)"
                match = re.findall(_markdown_image, doc.page_content)
                if match:
                    images.extend(match)
            assert len(images) >= 1

        if hasattr(parser, "password"):
            old_password = parser.password
            parser.password = "password"
            blob = Blob.from_path(LAYOUT_PARSER_PAPER_PASSWORD_PDF)
            doc_generator = parser.lazy_parse(blob)
            docs = list(doc_generator)
            assert len(docs)
            parser.password = old_password

    parser = parser_class(
        images_parser=image_parser,
        images_inner_format=images_inner_format,
        **kwargs,
    )
    _assert_with_parser(
        parser, splits_by_page=(kwargs.get("mode") in ("page", "paged"))
    )
    _std_assert_with_parser(parser)


@pytest.mark.parametrize(
    "extract_tables",
    ["markdown", "html", "csv", "text"],
)
@pytest.mark.parametrize(
    "parser_class,params",
    [
        (
            UnstructuredPDFParser,
            {"strategy": "hi_res"},
        ),
    ],
)
@pytest.mark.local
def test_parser_with_table(
    parser_class: Type,
    params: dict,
    extract_tables: Type_extract_tables,
) -> None:
    mode = "single"

    parser = parser_class(
        mode=mode,
        extract_tables=extract_tables,
        images_parser=EmptyImageBlobParser(),
        **params,
    )
    _std_assert_table_with_parser(extract_tables, parser)


def _std_assert_table_with_parser(extract_tables: str, parser: BaseBlobParser) -> None:
    """Standard tests to verify that the given parser works.

    Args:
        parser (BaseBlobParser): The parser to test.
    """
    blob = Blob.from_path(LAYOUT_PARSER_PAPER_PDF)
    doc_generator = parser.lazy_parse(blob)
    docs = list(doc_generator)
    tables = []
    for doc in docs:
        if extract_tables == "markdown":
            pattern = (
                r"(?s)("
                r"(?:(?:[^\n]*\|)\n)"
                r"(?:\|(?:\s?:?---*:?\s?\|)+)\n"
                r"(?:(?:[^\n]*\|)\n)+"
                r")"
            )
        elif extract_tables == "html":
            pattern = r"(?s)(<table[^>]*>(?:.*?)<\/table>)"
        elif extract_tables == "csv":
            pattern = (
                r"((?:(?:"
                r'(?:"(?:[^"]*(?:""[^"]*)*)"'
                r"|[^\n,]*),){2,}"
                r"(?:"
                r'(?:"(?:[^"]*(?:""[^"]*)*)"'
                r"|[^\n]*))\n){2,})"
            )
        else:
            pattern = None
        if pattern:
            matches = re.findall(pattern, doc.page_content)
            if matches:
                tables.extend(matches)
    if extract_tables != "text":
        assert len(tables) >= 1
    else:
        assert not len(tables)
