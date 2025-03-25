from pathlib import Path

import pytest

from langchain_unstructured import UnstructuredOrgModeLoader

EXAMPLE_DIRECTORY = Path(__file__).parent.parent / "examples"


@pytest.mark.local
def test_unstructured_org_mode_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLE_DIRECTORY / "README.org"
    loader = UnstructuredOrgModeLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1
