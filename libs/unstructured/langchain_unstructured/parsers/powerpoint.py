from typing import Any, Callable, Iterable, Optional

from langchain_core.documents.base import Blob

from langchain_unstructured.parsers.base import (
    DEFAULT_EXTRACT_TABLES,
    DEFAULT_MODE,
    DEFAULT_PAGE_DELIMITER,
    Type_all_mode,
    Type_extract_tables,
    UnstructuredClient,
    _UnstructuredDocumentParser,
)


class UnstructuredPowerPointParser(_UnstructuredDocumentParser):
    """parse `Microsoft PowerPoint` blob using `Unstructured`.

    Works with both .ppt and .pptx blob.
    You can run the parser in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Setup:
        Install ``langchain-unstructured[pptx,ppt]``.

        .. code-block:: bash

            pip install -U langchain-unstructured[pptx,ppt]

    Instantiate:

        .. code-block:: python
            from langchain_core.documents.base import Blob
            from langchain_unstructured.parsers import UnstructuredPowerPointParser

            blob=Blob.from_path("example.pptx")
            parser = UnstructuredPowerPointParser(
                mode="single",
            )
            docs= list(parser.lazy_parse(blob))
    """

    def __init__(
        self,
        *,
        partition_via_api: bool = False,
        post_processors: Optional[Iterable[Callable[[str], str]]] = None,
        mode: Type_all_mode = DEFAULT_MODE,
        extract_tables: Type_extract_tables = DEFAULT_EXTRACT_TABLES,
        pages_delimitor: str = DEFAULT_PAGE_DELIMITER,
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
                - "elements": Returns each identified element (e.g., title, text,
                  table) as a separate Document.
                Defaults to "single".
            extract_tables: Specifies how to handle tables found in the PDF.
                Can be "text" (extract text content), "html" (extract HTML
                representation), or "markdown" (convert to Markdown).
                Defaults to "markdown".
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
            from unstructured.file_utils.filetype import FileType
            from unstructured.partition.ppt import partition_ppt
            from unstructured.partition.pptx import partition_pptx
        except ImportError as e:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install 'langchain_unstructured[pptx]'`"
            ) from e

        mime_type_parser: dict[str, Callable] = {
            FileType.PPT.mime_type: partition_ppt,
            FileType.PPTX.mime_type: partition_pptx,
        }
        assert blob.mimetype in mime_type_parser, f"Invalide mime-type {blob.mimetype}"
        partition_fn = mime_type_parser[blob.mimetype]
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
