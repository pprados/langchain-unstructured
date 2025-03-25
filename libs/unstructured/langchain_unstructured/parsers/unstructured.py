from typing import Any, Callable, Iterable, Literal, Optional

from langchain_community.document_loaders.parsers import BaseImageBlobParser
from langchain_core.documents.base import Blob

from langchain_unstructured.parsers.base import (
    DEFAULT_EXTRACT_TABLES,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_PAGE_DELIMITER,
    DEFAULT_UNSTRUCTURED_MODE,
    Type_extract_tables,
    Type_image,
    UnstructuredClient,
    _UnstructuredDocumentParser,
)


class UnstructuredParser(_UnstructuredDocumentParser):
    """Parse files using `Unstructured`.

    Like other
    Unstructured parser, can be used in both
    "single" and "elements" mode. If you use the parser in "elements"
    mode, the TSV file will be a single Unstructured Table element.
    If you use the parser in "elements" mode, an HTML representation
    of the table will be available in the "text_as_html" key in the
    document metadata.

    Setup:
        Install ``langchain-unstructured[all_docs]``.

        .. code-block:: bash

            pip install -U langchain-unstructured[all_docs].

    Instantiate:

        .. code-block:: python
            from langchain_core.documents.base import Blob
            from langchain_unstructured.parsers import UnstructuredParser

            blob=Blob.from_path("example.odt")
            parser = UnstructuredParser(
                mode="single",
            )
            docs= list(parser.lazy_parse(blob))
    """

    def __init__(
        self,
        *,
        partition_via_api: bool = False,
        post_processors: Optional[Iterable[Callable[[str], str]]] = None,
        mode: Literal[
            "elements", "single", "paged", "page"
        ] = DEFAULT_UNSTRUCTURED_MODE,
        extract_tables: Type_extract_tables = DEFAULT_EXTRACT_TABLES,
        pages_delimitor: str = DEFAULT_PAGE_DELIMITER,
        images_parser: Optional[BaseImageBlobParser] = None,
        images_inner_format: Type_image = DEFAULT_IMAGE_FORMAT,
        # SDK parameters
        api_key: Optional[str] = None,
        client: Optional[UnstructuredClient] = None,
        url: Optional[str] = None,
        web_url: Optional[str] = None,
        **unstructured_kwargs: Any,
    ):
        """
        Args:
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
            **unstructured_kwargs: Additional keyword arguments to pass directly to the
                underlying Unstructured partitioning function (either local `partition`
                or the API endpoint). See the Unstructured documentation for available
                options (e.g., `strategy`, `ocr_languages`, `encoding`).
        """
        super().__init__(
            partition_via_api=partition_via_api,
            post_processors=post_processors,
            mode=mode,
            images_parser=images_parser,
            images_inner_format=images_inner_format,
            extract_tables=extract_tables,
            pages_delimitor=pages_delimitor,
            api_key=api_key,
            client=client,
            url=url,
            web_url=web_url,
            **unstructured_kwargs,
        )

    def _get_elements_via_local(self, blob: Blob) -> list:
        try:
            from unstructured.partition.auto import partition
        except ImportError as e:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install 'langchain_unstructured[all_docs]' or some file formats`"
            ) from e

        partition_fn = partition
        if blob.data is None and blob.path:
            return partition_fn(
                filename=str(blob.path), file=None, **self.unstructured_kwargs
            )
        else:
            with blob.as_bytes_io() as file:
                return partition_fn(
                    filename=None,
                    file=file,
                    content_type=blob.mimetype,
                    **self.unstructured_kwargs,
                )
