"""Loader that uses unstructured to load files."""
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from pathlib import PurePath
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

from langchain_unstructured.base import UnstructuredClient, _UnstructuredBaseLoader
from langchain_unstructured.parsers.base import (
    DEFAULT_EXTRACT_TABLES,
    DEFAULT_PAGE_DELIMITER,
    DEFAULT_UNSTRUCTURED_MODE,
    Type_all_mode,
    Type_extract_tables, DEFAULT_KEEP_HEADER_FOOTER,
)
from langchain_unstructured.parsers.unstructured import UnstructuredParser


class UnstructuredLoader(_UnstructuredBaseLoader):
    """Load files using `Unstructured`.

    Like other
    Unstructured loaders, can be used in both
    "single" and "elements" mode. If you use the loader in "elements"
    mode, the TSV file will be a single Unstructured Table element.
    If you use the loader in "elements" mode, an HTML representation
    of the table will be available in the "text_as_html" key in the
    document metadata.

    Setup:
        Install ``langchain-unstructured[all_docs]``.

        .. code-block:: bash

            pip install -U langchain-unstructured[all_docs].

    Instantiate:
        .. code-block:: python

            from langchain_community import UnstructuredTSVLoader

            loader = UnstructuredTSVLoader(
                "example.tsv",
                mode="single",
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
    """

    def __init__(
            self,
            file_path: Union[
                Sequence[str], Sequence[PurePath], str, PurePath, None] = None,
            *,
            file: Optional[IO[bytes]] = None,
            partition_via_api: bool = False,
            post_processors: Optional[Iterable[Callable[[str], str]]] = None,
            mode: Type_all_mode = DEFAULT_UNSTRUCTURED_MODE,
            extract_tables: Type_extract_tables = DEFAULT_EXTRACT_TABLES,
            pages_delimitor: str = DEFAULT_PAGE_DELIMITER,
            keep_header_footer:bool = DEFAULT_KEEP_HEADER_FOOTER,
            # SDK parameters
            api_key: Optional[str] = None,
            client: Optional[UnstructuredClient] = None,
            url: Optional[str] = None,
            web_url: Optional[str] = None,
            **unstructured_kwargs: Any,
    ):
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
            url: URL for API. Can be passed directly or set via the
                `UNSTRUCTURED_API_KEY` environment variable.
            post_processors: An optional iterable of functions, each taking a string
                (element text) and returning a string. These functions are applied
                sequentially to the text content of each extracted element after
                partitioning.
            api_key: The API key for the Unstructured API. Required if
                `partition_via_api` is True and `client` is not provided. Can be
                passed directly or set via the `UNSTRUCTURED_API_KEY` environment
                variable.
            client: An optional pre-configured `UnstructuredClient` instance to use
                for API calls. If provided, `api_key` is ignored.
            password: The password to use for decrypting password-protected PDF files.
            **unstructured_kwargs: Additional keyword arguments to pass directly to the
                underlying Unstructured partitioning function (either local `partition`
                or the API endpoint). See the Unstructured documentation for available
                options (e.g., `strategy`, `ocr_languages`, `encoding`).
        """
        # This class is the only one to accept a file_path with file list. This is
        # not shared with other loaders. We've implemented what's necessary to
        # maintain compatibility, even though we don't share this approach. We believe
        # it's up to the consumer of the class to loop through the files to be
        # processed.
        super().__init__(
            parser=UnstructuredParser(
                partition_via_api=partition_via_api,
                post_processors=post_processors,
                mode=mode,
                extract_tables=extract_tables,
                pages_delimitor=pages_delimitor,
                keep_header_footer=keep_header_footer,
                api_key=api_key,
                client=client,
                url=url,
                **unstructured_kwargs,
            ),
            file_path=(
                file_path if isinstance(file_path, (str, PurePath, type(None)))
                else file_path[0]
            ),
            file=file,
            web_url=web_url,
            **unstructured_kwargs,
        )
        self._file_paths = file_path

    def lazy_load(self) -> Iterator[Document]:
        if not isinstance(self._file_paths, (str, PurePath, type(None))):
            for file_path in self._file_paths:
                blob = Blob.from_path(str(file_path), guess_type=True)
                yield from self._lazy_load_blob(blob)
        else:
            yield from super().lazy_load()
