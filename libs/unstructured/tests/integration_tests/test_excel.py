from pathlib import Path

import pytest

from langchain_unstructured import UnstructuredExcelLoader

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"


@pytest.mark.local
def test_unstructured_excel_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLE_DIRECTORY / "stanley-cups.xlsx"
    loader = UnstructuredExcelLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1
