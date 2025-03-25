import mimetypes
from pathlib import Path
from typing import Any, Callable
from unittest import mock
from unittest.mock import Mock, mock_open, patch

import pytest
from langchain_core.documents.base import Blob
from langchain_unstructured import (
    UnstructuredLoader,
)
from langchain_unstructured.base import (
    _UnstructuredBaseLoader,
)
from unstructured.documents.elements import Text, Element

from libs.unstructured.langchain_unstructured.parsers.base import \
    _UnstructuredDocumentParser

mimetypes.init()

EXAMPLE_DOCS_DIRECTORY = (
        Path(__file__).parent.parent.parent.parent.parent
        / "community/tests/integration_tests/examples/"
)


# --- UnstructuredLoader.__init__() ---


def test_it_initializes_with_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNSTRUCTURED_API_KEY", raising=False)

    loader = UnstructuredLoader(file_path="dummy_path")

    assert loader.file_path == "dummy_path"
    assert loader.file is None
    assert loader.parser.partition_via_api is False
    assert loader.parser.post_processors is None
    if loader.parser.extract_tables != "text":
        assert loader.parser.unstructured_kwargs.get("infer_table_structure") is True
    else:
        assert loader.parser.unstructured_kwargs == {}
    assert loader.parser.client is None


def test_it_initializes_with_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "FAKE_API_KEY")

    loader = UnstructuredLoader(file_path="dummy_path.txt")
    assert loader.parser.api_key == "FAKE_API_KEY"


def test_it_gets_content_from_file() -> None:
    with mock.patch("langchain_unstructured.base._UnstructuredBaseLoader._parse_blob"):
        mock_file = Mock()
        mock_file.read.return_value = b"content from file"
        loader = UnstructuredLoader(
            client=Mock(), file=mock_file, metadata_filename="fake.txt"
        )

        loader.load()
        blob = _UnstructuredBaseLoader._parse_blob.call_args[0][0]  # type: ignore
        assert blob.data == b"content from file"
        assert blob.path is None
        assert blob.mimetype == "text/plain"


@patch("builtins.open", new_callable=mock_open, read_data=b"content from file_path")
def test_it_gets_content_from_file_path(mock_file: Mock) -> None:
    with mock.patch("langchain_unstructured.base._UnstructuredBaseLoader._parse_blob"):
        loader = UnstructuredLoader(client=Mock(), file_path="dummy_path.txt")

        loader.load()
        blob = _UnstructuredBaseLoader._parse_blob.call_args[0][0]  # type: ignore
        assert blob.data is None
        assert blob.path == "dummy_path.txt"
        assert blob.mimetype == "text/plain"


def test_it_raises_value_error_without_file_or_file_path() -> None:
    with pytest.raises(ValueError) as e:
        UnstructuredLoader(
            client=Mock(),
        )

    assert str(e.value) == "One of file_path, file, or web_url must be provided."


def test_it_raises_value_error_with_file_and_file_path() -> None:
    with pytest.raises(ValueError) as e:
        UnstructuredLoader(
            client=Mock(),
            file=Mock(),
            file_path="dummy_path.txt",
            web_url="http://dummy.org",
        )

    assert str(e.value) == (
        "file_path, file or web_url cannot be defined simultaneously."
    )


def test_it_calls_elements_via_api_with_valid_args() -> None:
    with patch.object(
            _UnstructuredDocumentParser,
            "_get_elements_via_api",
            return_value=[{"element": "data"}]
    ) as mock_get_elements_via_api:
        class X(_UnstructuredDocumentParser):
            def _get_elements_via_local(self, blob: Blob) -> list[Element]:
                return []

        parser = X(
            mode="single",
            client=Mock(),
            # Minimum required args for self._elements_via_api to be called:
            partition_via_api=True,
        )

        result = parser._get_elements_json(Blob.from_data(data=""))

    mock_get_elements_via_api.assert_called_once()
    assert result == [{"element": "data"}]


@patch.object(_UnstructuredDocumentParser, "_convert_elements_to_dicts")
def test_it_partitions_locally_by_default(mock_convert_elements_to_dicts: Mock) -> None:
    mock_convert_elements_to_dicts.return_value = [{}]
    # Minimum required args for self._elements_via_api to be called:
    class X(_UnstructuredDocumentParser):
        def _get_elements_via_local(self, blob: Blob) -> list[Element]:
            return [{"element": "data"}]

    parser = X(
        mode="single",
        client=Mock(),
        # Minimum required args for self._elements_via_api to be called:
    )

    result = parser._get_elements_json(Blob.from_data(data=""))
    assert result == [{}]

# -- fixtures -------------------------------


@pytest.fixture()
def get_post_processor() -> Callable[[str], str]:
    def append_the_end(text: str) -> str:
        return text + "THE END!"

    return append_the_end


@pytest.fixture()
def fake_json_response() -> list[dict[str, Any]]:
    return [
        {
            "type": "Title",
            "element_id": "b7f58c2fd9c15949a55a62eb84e39575",
            "text": "LayoutParser: A Uniﬁed Toolkit for Deep Learning Based Document"
                    "Image Analysis",
            "metadata": {
                "languages": ["eng"],
                "page_number": 1,
                "filename": "layout-parser-paper.pdf",
                "filetype": "application/pdf",
            },
        },
        {
            "type": "UncategorizedText",
            "element_id": "e1c4facddf1f2eb1d0db5be34ad0de18",
            "text": "1 2 0 2",
            "metadata": {
                "languages": ["eng"],
                "page_number": 1,
                "parent_id": "b7f58c2fd9c15949a55a62eb84e39575",
                "filename": "layout-parser-paper.pdf",
                "filetype": "application/pdf",
            },
        },
    ]


@pytest.fixture()
def fake_multiple_docs_json_response() -> list[dict[str, Any]]:
    return [
        {
            "type": "Title",
            "element_id": "b7f58c2fd9c15949a55a62eb84e39575",
            "text": "LayoutParser: A Uniﬁed Toolkit for Deep Learning Based Document"
                    " Image Analysis",
            "metadata": {
                "languages": ["eng"],
                "page_number": 1,
                "filename": "layout-parser-paper.pdf",
                "filetype": "application/pdf",
            },
        },
        {
            "type": "NarrativeText",
            "element_id": "3c4ac9e7f55f1e3dbd87d3a9364642fe",
            "text": "6/29/23, 12:16\u202fam - User 4: This message was deleted",
            "metadata": {
                "filename": "whatsapp_chat.txt",
                "languages": ["eng"],
                "filetype": "text/plain",
            },
        },
    ]
