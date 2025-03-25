import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents.base import Blob
from unstructured.file_utils.filetype import FileType

from langchain_unstructured.parsers import UnstructuredCSVParser
from tests.integration_tests.parsers import couple_of_blob

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent.parent / "examples"


@pytest.mark.parametrize(
    "blob",
    couple_of_blob(EXAMPLE_DIRECTORY / "stanley-cups.csv", FileType.CSV.mime_type),
)
@pytest.mark.parametrize(
    "kwargs,len_docs",
    [
        ({}, 1),
        ({"mode": "single"}, 1),
        ({"mode": "elements"}, 1),
        ({"mode": "elements", "extract_tables": "html"}, 1),
    ],
)
@pytest.mark.local
def test_unstructured_csv_parser(kwargs: Any, len_docs: int, blob: Blob) -> None:
    parser = UnstructuredCSVParser(**kwargs)
    docs = list(parser.lazy_parse(blob))

    assert len(docs) == len_docs


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
@pytest.mark.parametrize(
    "blob",
    couple_of_blob(EXAMPLE_DIRECTORY / "stanley-cups.csv", FileType.CSV.mime_type),
)
def test_unstructured_csv_parser_via_api(blob: Blob) -> None:
    """Test unstructured loader."""
    parser = UnstructuredCSVParser(mode="single", partition_via_api=True)
    docs = list(parser.lazy_parse(blob))
    assert len(docs) == 1
