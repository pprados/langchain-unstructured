"""Unstructured document loader."""

from __future__ import annotations

import logging
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.images import (
    BaseImageBlobParser,
)
from langchain_core.documents import Document
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal, Optional,
)
from typing_extensions import IO, TypeAlias

from langchain_unstructured.base import Blob_from_url, UnstructuredClient, \
    _UnstructuredBaseLoader
from langchain_unstructured.parsers import UnstructuredPDFParser
from langchain_unstructured.parsers.base import (
    DEFAULT_EXTRACT_TABLES,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MODE,
    DEFAULT_PAGE_DELIMITER,
    Type_extract_tables,
    Type_file_path,
    Type_image, DEFAULT_KEEP_HEADER_FOOTER,
)

Element: TypeAlias = Any

logger = logging.getLogger(__file__)

_DEFAULT_URL = "https://api.unstructuredapp.io/general/v0/general"


class UnstructuredPDFLoader(_UnstructuredBaseLoader):
    """Load PDF file with Unstructured.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Setup:
        Install ``langchain-unstructured[pdf]``.

        .. code-block:: bash

            pip install -U langchain-unstructured[pdf]

    Instantiate:
        .. code-block:: python

            from langchain_community import UnstructuredPDFLoader

            loader = UnstructuredPDFLoader(
                "example.pdf",
                mode="single",
                strategy="auto",  # FIXME: ajout images
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)
    """  # noqa: E501à

    def __init__(
            self,
            file_path: Type_file_path = None,
            *,
            file: Optional[IO[bytes]] = None,
            web_url: Optional[str] = None,
            headers: Optional[dict] = None,
            mode: Literal["single", "page", "elements"] = DEFAULT_MODE,
            pages_delimitor: str = DEFAULT_PAGE_DELIMITER,
            images_parser: BaseImageBlobParser | None = None,
            images_inner_format: Type_image = DEFAULT_IMAGE_FORMAT,
            extract_tables: Type_extract_tables = DEFAULT_EXTRACT_TABLES,
            keep_header_footer: bool = DEFAULT_KEEP_HEADER_FOOTER,
            partition_via_api: bool = False,
            post_processors: Iterable[Callable[[str], str]] | None = None,
            # SDK parameters
            api_key: Optional[str] = None,
            client: UnstructuredClient | None = None,
            password: Optional[str] = None,
            **unstructured_kwargs: Any,
    ) -> None:
        """Initialize loader.

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
            mode: The mode for partitioning the PDF. Options are:
                - "single": Returns the entire document content as a single Document.
                - "page": Splits the document by page, returning one Document per page.
                - "elements": Returns each identified element (e.g., title, text,
                  table) as a separate Document.
                Defaults to "single".
            pages_delimitor: The string delimiter used to join page contents when
                `mode` is "page". Defaults to "\n\f\n".
            images_parser: An optional parser (subclass of `BaseImageBlobParser`) to
                process images extracted from the PDF. If provided, image elements
                will be parsed using this parser.
            images_inner_format: The desired format inject the text from the image
                (e.g., "text", "markdown-img" or "html-img").
                Defaults to "markdown-img".
            extract_tables: Specifies how to handle tables found in the PDF.
                Can be "text" (extract text content), "html" (extract HTML
                representation), or "markdown" (convert to Markdown).
                Defaults to "markdown".
            keep_header_footer: Keep the header and footer. Default to False.
            partition_via_api: If True, uses the Unstructured API for partitioning.
                Requires setting `api_key` or having the `UNSTRUCTURED_API_KEY`
                environment variable set. Defaults to False (uses local partitioning).
            post_processors: An optional iterable of functions, each taking a string
                (element text) and returning a string. These functions are applied
                sequentially to the text content of each extracted element after
                partitioning.
            api_key: The API key for the Unstructured API. Required if
                `partition_via_api` is True and `client` is not provided. Can be
                passed directly or set via the `UNSTRUCTURED_API_KEY` environment
                variable.
            url: URL for API. Can be passed directly or set via the
                `UNSTRUCTURED_API_KEY` environment variable.
            client: An optional pre-configured `UnstructuredClient` instance to use
                for API calls. If provided, `api_key` is ignored.
            password: The password to use for decrypting password-protected PDF files.
            **unstructured_kwargs: Additional keyword arguments to pass directly to the
                underlying Unstructured partitioning function (either local `partition`
                or the API endpoint). See the Unstructured documentation for available
                options (e.g., `strategy`, `ocr_languages`, `encoding`).
        """
        super().__init__(
            parser=UnstructuredPDFParser(
                mode=mode,  # typing: ignore
                pages_delimitor=pages_delimitor,
                images_parser=images_parser,
                images_inner_format=images_inner_format,
                extract_tables=extract_tables,
                keep_header_footer=keep_header_footer,
                client=client,
                partition_via_api=partition_via_api,
                post_processors=post_processors,
                password=password,
                api_key=api_key,
                **unstructured_kwargs,
            ),
            file_path=file_path,
            file=file,
            web_url=web_url,
            headers=headers,
            **unstructured_kwargs,
        )
