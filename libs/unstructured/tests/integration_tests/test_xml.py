from pathlib import Path

import pytest

from langchain_unstructured import UnstructuredXMLLoader

EXAMPLE_DIRECTORY = Path(__file__).parent.parent / "examples"


@pytest.mark.local
def test_unstructured_xml_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLE_DIRECTORY / "factbook.xml"
    loader = UnstructuredXMLLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1
