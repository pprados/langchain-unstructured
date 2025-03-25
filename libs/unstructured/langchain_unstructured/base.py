"""Unstructured document loader."""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from abc import ABC
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    cast,
)

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import _purge_metadata
from langchain_community.document_loaders.unstructured import (
    validate_unstructured_version,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from typing_extensions import TypeAlias

from langchain_unstructured.parsers.base import (
    Type_file_path,
    _UnstructuredDocumentParser,
)

if TYPE_CHECKING:
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import operations  # type:ignore[attr-defined]
else:
    UnstructuredClient = Any

Element: TypeAlias = Any

logger = logging.getLogger(__file__)

_DEFAULT_URL = "https://api.unstructuredapp.io/general/v0/general"


def Blob_from_url(web_url: str, headers: Optional[dict]) -> Blob:
    """Fetches content from a web URL and returns it as a Blob object.

    Uses the `requests` library to perform an HTTP GET request to the specified URL.
    The content, headers, mimetype, encoding, and final URL from the response
    are used to populate the returned Blob object.

    Args:
        web_url: The URL of the resource to fetch.
        headers: An optional dictionary of HTTP headers to include in the GET request.

    Returns:
        A Blob object containing the fetched data. The Blob's attributes
     """
    import requests

    with requests.get(web_url, headers=headers) as response:
        return Blob(
            data=response.content,
            metadata=dict(response.headers),
            mimetype=str(response.headers["content-type"]),
            encoding=str(response.encoding),
            path=response.url,
        )


class _UnstructuredBaseLoader(BaseLoader, ABC):
    """Base Loader that uses `Unstructured`."""

    def __init__(
        self,
        *,
        parser: _UnstructuredDocumentParser,
        file_path: Type_file_path = None,
        file: Optional[IO[bytes]] = None,
        web_url: Optional[str] = None,
        headers: Optional[dict] = None,
        **unstructured_kwargs: Any,
    ):
        """unstructured base loader.

        Args:
            file_path: The path to the PDF file to load. Can be a single path (string
                or Path object). Mutually exclusive with `file`
                and `web_url`.
            file: A file-like object opened in binary read mode (`IO[bytes]`).
                Mutually exclusive with `file_path` and `web_url`.
            web_url: A URL from which to download the PDF file. Mutually exclusive
                with `file_path` and `file`.
            headers: A dictionary of headers to use when fetching the PDF from a
                `web_url`.
            **unstructured_kwargs: Additional keyword arguments to pass directly to the
                underlying Unstructured partitioning function (either local `partition`
                or the API endpoint). See the Unstructured documentation for available
                options (e.g., `strategy`, `ocr_languages`, `encoding`).
        """
        validate_unstructured_version(min_unstructured_version="0.6.8")
        p = sum(1 for _ in (file_path, file, web_url) if _)
        if not p:
            raise ValueError("One of file_path, file, or web_url must be provided.")
        if p > 1:
            raise ValueError(
                "file_path, file or web_url cannot be defined simultaneously."
            )
        self.parser = parser
        self.file_path = file_path
        self.file = file
        self.web_url = web_url
        self.headers = headers
        self.filename = unstructured_kwargs.get("metadata_filename")

    def _lazy_load_blob(self, blob: Blob) -> Iterator[Document]:
        # Special method to manage the file_path with list, only for
        # UnstructuredLoader.
        if not blob.mimetype:
            from unstructured.file_utils.filetype import detect_filetype

            if not self.file_path:
                assert self.filename is not None
                mime_type = mimetypes.guess_type(self.filename)[0]
            else:
                mime_type = detect_filetype(file_path=str(self.file_path)).mime_type
            if not mime_type:
                mime_type = "text/plain"
            blob = Blob(
                id=blob.id,
                metadata=blob.metadata,
                data=blob.data,
                encoding=blob.encoding,
                mimetype=mime_type,
                path=blob.path,
            )
        return self._parse_blob(blob)

    def lazy_load(self) -> Iterator[Document]:
        # Generate a blob, and parse it.
        if self.web_url:
            blob = Blob_from_url(self.web_url, self.headers)
        elif self.file:
            blob = Blob.from_data(self.file.read())
        else:
            blob = Blob.from_path(str(self.file_path), guess_type=True)
        yield from self._lazy_load_blob(blob)

    def _parse_blob(self, blob: Blob) -> Iterator[Document]:
        for doc in self.parser.lazy_parse(blob):
            if self.web_url:
                doc.metadata["url"] = self.web_url
            yield doc

