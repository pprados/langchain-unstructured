from pathlib import Path

import pytest

from langchain_unstructured import UnstructuredODTLoader

EXAMPLE_DIRECTORY = Path(__file__).parent.parent / "examples"


@pytest.mark.local
def test_unstructured_odt_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLE_DIRECTORY / "README.odt"
    loader = UnstructuredODTLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1
