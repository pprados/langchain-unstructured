"""Unstructured document loader."""

from __future__ import annotations

import logging
import sys
from tempfile import TemporaryDirectory
from typing import (
    Any,
    BinaryIO,
    Callable,
    Iterable,
    Literal,
)

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.images import (
    BaseImageBlobParser,
)
from langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser,
)
from typing_extensions import TypeAlias

from langchain_unstructured.parsers.base import (
    DEFAULT_EXTRACT_TABLES,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_KEEP_HEADER_FOOTER,
    DEFAULT_MODE,
    DEFAULT_PAGE_DELIMITER,
    Type_extract_tables,
    Type_image,
    UnstructuredClient,
    _UnstructuredDocumentParser,
    satisfies_min_pdfminer_six_version,
)

Element: TypeAlias = Any

logger = logging.getLogger(__file__)

_DEFAULT_URL = "https://api.unstructuredapp.io/general/v0/general"


class UnstructuredPDFParser(_UnstructuredDocumentParser):
    """Parser PDF blob with Unstructured.

        You can run the parser in one of two modes: "single" and "elements".
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
            from langchain_core.documents.base import Blob
            from langchain_unstructured.parsers import UnstructuredPDFParser

            blob=Blob.from_path("example.pdf")
            parser = UnstructuredPDFParser(
                mode="single",
            )
            docs= list(parser.lazy_parse(blob))
        """
    _warn_extract_tables = False

    def __init__(
        self,
        *,
        password: str | None = None,
        mode: Literal["single", "page", "paged", "elements"] = DEFAULT_MODE,
        pages_delimitor: str = DEFAULT_PAGE_DELIMITER,
        images_parser: BaseImageBlobParser | None = None,
        images_inner_format: Type_image = DEFAULT_IMAGE_FORMAT,
        extract_tables: Type_extract_tables = DEFAULT_EXTRACT_TABLES,
        keep_header_footer: bool = DEFAULT_KEEP_HEADER_FOOTER,
        partition_via_api: bool = False,
        post_processors: Iterable[Callable[[str], str]] | None = None,
        # SDK parameters
        api_key: str | None = None,
        client: UnstructuredClient | None = None,
        url: str | None = None,
        web_url: str | None = None,
        **unstructured_kwargs: Any,
    ) -> None:
        """Initialize the parser.
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
        if not satisfies_min_pdfminer_six_version("20250327"):
            raise ValueError('pdfminer.six>="20250327" is required in this loader.')
        if unstructured_kwargs.get("strategy") == "ocr_only" and images_parser:
            logger.warning("images_parser is not supported with strategy='ocr_only")
        if unstructured_kwargs.get("strategy") != "hi_res" and extract_tables:
            if not UnstructuredPDFParser._warn_extract_tables:
                UnstructuredPDFParser._warn_extract_tables = True
                logger.warning(
                    "extract_tables is not supported with strategy!='hi_res'"
                )
            extract_tables:Type_extract_tables = "text"
        super().__init__(
            pages_delimitor=pages_delimitor,
            images_parser=images_parser,
            images_inner_format=images_inner_format,
            extract_tables=extract_tables,
            keep_header_footer=keep_header_footer,
            partition_via_api=partition_via_api,
            post_processors=post_processors,
            mode=mode,
            api_key=api_key,
            client=client,
            url=url,
            **unstructured_kwargs,
        )

        self.password = password
        self.tmp_dir = None
        if self.images_parser:
            if partition_via_api:
                logger.warning("images_parser is not supported with partition_via_api")
            else:
                unstructured_kwargs["extract_images_in_pdf"] = True
                if sys.version_info >= (3,10):
                    kwargs={
                        "ignore_cleanup_errors": True
                    }
                else:
                    kwargs={}
                self.tmp_dir = TemporaryDirectory(
                    **kwargs
                )
                if "extract_image_block_output_dir" not in unstructured_kwargs:
                    unstructured_kwargs["extract_image_block_output_dir"] = (
                        self.tmp_dir.name
                    )

    def _get_elements_via_local(self, blob: Blob) -> list:
        try:
            from unstructured.file_utils.filetype import FileType
            from unstructured.partition.pdf import partition_pdf
        except ImportError as e:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install 'langchain_unstructured[pdf]'`"
            ) from e
        assert blob.mimetype == FileType.PDF.mime_type, (
            f"Invalide mime-type {blob.mimetype}"
        )

        unstructured_kwargs = self.unstructured_kwargs.copy()
        if not self.partition_via_api:
            unstructured_kwargs["metadata_filename"] = blob.path or blob.metadata.get(
                "source"
            )
        if self.extract_tables != "text":
            unstructured_kwargs["infer_table_structure"] = True
        if self.mode != "elements":
            unstructured_kwargs["include_page_breaks"] = True
        partition_fn = partition_pdf
        if blob.data is None and blob.path:
            return partition_fn(
                filename=str(blob.path),
                file=None,
                password=self.password,
                **unstructured_kwargs,
            )
        else:
            with blob.as_bytes_io() as file:
                return partition_fn(
                    filename=None,
                    file=file,
                    content_type=blob.mimetype,
                    password=self.password,
                    **unstructured_kwargs,
                )

    def _get_metadata(
        self,
        fp: BinaryIO,
        blob: Blob,
    ) -> dict[str, Any]:
        """
        Extract metadata from a PDF file.

        Args:
            fp: The file pointer to the PDF file.
            blob: The current blob.

        Returns:
            Metadata of the PDF file.
        """
        from pdfminer.pdfpage import PDFDocument, PDFPage, PDFParser

        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        doc = PDFDocument(parser, password=self.password or "", caching=True)
        metadata = {}

        for info in doc.info:
            metadata.update(info)
        for k, v in metadata.items():
            try:
                metadata[k] = PDFMinerParser.resolve_and_decode(v)
            except Exception as e:  # pragma: nocover
                # This metadata value could not be parsed. Instead of failing the PDF
                # read, treat it as a warning only if `strict_metadata=False`.
                logger.warning(
                    '[WARNING] Metadata key "%s" could not be parsed due to '
                    "exception: %s",
                    k,
                    str(e),
                )

        # Count number of pages.
        metadata["total_pages"] = len(list(PDFPage.create_pages(doc)))

        return metadata
