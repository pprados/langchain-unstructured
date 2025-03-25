import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents.base import Blob
from unstructured.file_utils.filetype import FileType

from langchain_unstructured.parsers import UnstructuredParser
from langchain_unstructured.parsers.base import DEFAULT_UNSTRUCTURED_MODE
from tests.integration_tests.parsers import couple_of_blob

EXAMPLE_DIRECTORY = Path(__file__).parent.parent.parent / "examples"


@pytest.mark.parametrize(
    "blob",
    couple_of_blob(EXAMPLE_DIRECTORY / "README.html", FileType.HTML.mime_type),
)
@pytest.mark.parametrize(
    "kwargs,doc_lens",
    [
        ({}, 10 if DEFAULT_UNSTRUCTURED_MODE == "elements" else 1),
        ({"mode": "single"}, 1),
        ({"mode": "elements"}, 10),
        ({"mode": "page", "extra_tables": "html"}, 1),
    ],
)
@pytest.mark.local
def test_unstructured_parser(kwargs: Any, doc_lens: int, blob: Blob) -> None:
    """Test unstructured loader."""
    parser = UnstructuredParser(**kwargs)
    docs = list(parser.lazy_parse(blob))

    assert len(docs) == doc_lens


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
@pytest.mark.parametrize(
    "blob",
    couple_of_blob(EXAMPLE_DIRECTORY / "image.png", FileType.PNG.mime_type),
)
def test_unstructured_parser_via_api(blob: Blob) -> None:
    """Test unstructured loader."""
    parser = UnstructuredParser(mode="single", partition_via_api=True)
    docs = list(parser.lazy_parse(blob))

    assert len(docs) == 1
