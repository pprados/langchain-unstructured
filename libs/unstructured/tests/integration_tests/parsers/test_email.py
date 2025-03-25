import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents.base import Blob
from unstructured.file_utils.filetype import FileType

from langchain_unstructured.parsers import UnstructuredEmailParser
from tests.integration_tests.parsers import couple_of_blob

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent.parent / "examples"


@pytest.mark.parametrize(
    "blob",
    couple_of_blob(
        EXAMPLE_DIRECTORY / "fake-email-attachment.eml", FileType.EML.mime_type
    ),
)
@pytest.mark.parametrize(
    "kwargs,len_docs",
    [
        ({}, 1),
        ({"mode": "single"}, 1),
        ({"mode": "elements"}, 9),
        ({"mode": "single", "process_attachments": True}, 1),
        ({"mode": "elements", "process_attachments": True}, 9),
    ],
)
@pytest.mark.local
def test_unstructured_email_with_attachments_parser(
    kwargs: Any, len_docs: int, blob: Blob
) -> None:
    """Test unstructured loader."""
    parser = UnstructuredEmailParser(**kwargs)
    docs = list(parser.lazy_parse(blob))
    assert len(docs) == len_docs


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
@pytest.mark.parametrize(
    "blob",
    couple_of_blob(
        EXAMPLE_DIRECTORY / "fake-email-attachment.eml", FileType.EML.mime_type
    ),
)
def test_unstructured_email_with_attachments_parser_via_api(blob: Blob) -> None:
    """Test unstructured loader."""
    parser = UnstructuredEmailParser(mode="single", partition_via_api=True)
    docs = list(parser.lazy_parse(blob))
    assert len(docs) == 1


@pytest.mark.parametrize(
    "blob", couple_of_blob(EXAMPLE_DIRECTORY / "hello.msg", FileType.MSG.mime_type)
)
@pytest.mark.local
def test_unstructured_email_without_attachments_parser(blob: Blob) -> None:
    """Test unstructured loader."""
    parser = UnstructuredEmailParser(mode="single")
    docs = list(parser.lazy_parse(blob))
    assert len(docs) == 1


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
@pytest.mark.parametrize(
    "blob", couple_of_blob(EXAMPLE_DIRECTORY / "hello.msg", FileType.MSG.mime_type)
)
@pytest.mark.local
def test_unstructured_email_without_attachments_parser_via_api(blob: Blob) -> None:
    """Test unstructured loader."""
    parser = UnstructuredEmailParser(mode="single", partition_via_api=True)
    docs = list(parser.lazy_parse(blob))
    assert len(docs) == 1
