from typing import Any, Callable, Iterable, Optional

from langchain_community.document_loaders.unstructured import (
    validate_unstructured_version,
)
from langchain_core.documents.base import Blob

from langchain_unstructured.parsers.base import (
    DEFAULT_EXTRACT_TABLES,
    DEFAULT_MODE,
    Type_extract_tables,
    Type_simple_mode,
    UnstructuredClient,
    _UnstructuredDocumentParser,
)


class UnstructuredEmailParser(_UnstructuredDocumentParser):
    """Parse email blob using `Unstructured`.

    Works with both
    .eml and .msg blob. You can process attachments in addition to the
    e-mail message itself by passing process_attachments=True into the
    constructor for the parser. By default, attachments will be processed
    with the unstructured partition function. If you already know the document
    types of the attachments, you can specify another partitioning function
    with the attachment partitioner kwarg.

    Setup:
        Install ``langchain-unstructured[email]``.

        .. code-block:: bash

            pip install -U langchain-unstructured[email]

    Instantiate:

        .. code-block:: python
            from langchain_core.documents.base import Blob
            from langchain_unstructured.parsers import UnstructuredEmailParser

            blob=Blob.from_path("example.eml")
            parser = UnstructuredEmailParser(
                mode="single",
            )
            docs= list(parser.lazy_parse(blob))
    """

    def __init__(
        self,
        *,
        partition_via_api: bool = False,
        post_processors: Optional[Iterable[Callable[[str], str]]] = None,
        mode: Type_simple_mode = DEFAULT_MODE,
        extract_tables: Type_extract_tables = DEFAULT_EXTRACT_TABLES,
        process_attachments: bool = False,
        attachment_partitioner: Optional[Callable] = None,
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
        validate_unstructured_version(min_unstructured_version="0.6.8")
        process_attachments = process_attachments
        attachment_partitioner = attachment_partitioner

        if process_attachments and attachment_partitioner is None:
            from unstructured.partition.auto import partition

            attachment_partitioner = partition

        if process_attachments:
            unstructured_kwargs["process_attachments"] = True
        if attachment_partitioner:
            unstructured_kwargs["attachment_partitioner"] = True

        super().__init__(
            partition_via_api=partition_via_api,
            post_processors=post_processors,
            mode=mode,
            extract_tables=extract_tables,
            api_key=api_key,
            client=client,
            url=url,
            web_url=web_url,
            **unstructured_kwargs,
        )

    def _get_elements_via_local(self, blob: Blob) -> list:
        try:
            from unstructured.file_utils.filetype import FileType
            from unstructured.partition.email import partition_email
            from unstructured.partition.msg import partition_msg
        except ImportError as e:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install 'langchain_unstructured'`"
            ) from e
        mime_type_parser: dict[str, Callable] = {
            FileType.EML.mime_type: partition_email,
            FileType.MSG.mime_type: partition_msg,
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
