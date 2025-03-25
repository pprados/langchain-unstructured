from pathlib import Path

import pytest

from langchain_unstructured.html import UnstructuredHTMLLoader

EXAMPLE_DIRECTORY = Path(__file__).parent.parent / "examples"


@pytest.mark.local
def test_unstructured_html_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLE_DIRECTORY / "README.html"
    loader = UnstructuredHTMLLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1
