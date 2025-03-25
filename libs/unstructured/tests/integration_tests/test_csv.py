from pathlib import Path

import pytest

from langchain_unstructured import UnstructuredCSVLoader

EXAMPLE_DIRECTORY = Path(__file__).parent.parent / "examples"


@pytest.mark.local
def test_unstructured_csv_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLE_DIRECTORY / "stanley-cups.csv"
    loader = UnstructuredCSVLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1
