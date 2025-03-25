from pathlib import Path

import pytest

from langchain_unstructured.image import UnstructuredImageLoader

EXAMPLE_DIRECTORY = Path(__file__).parent.parent / "examples"


@pytest.mark.local
def test_unstructured_image_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLE_DIRECTORY / "image.png"
    loader = UnstructuredImageLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1

    loader = UnstructuredImageLoader(file_path,mode="elements", strategy="fast",)
    docs = loader.load()

    assert len(docs) == 1
