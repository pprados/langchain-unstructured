"""Unstructured document loader."""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import abstractmethod
from copy import deepcopy
from pathlib import PurePath
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Union,
    cast, Optional,
)

from bs4 import BeautifulSoup
from langchain_community.document_loaders.parsers.images import (
    BaseImageBlobParser,
)
from langchain_community.document_loaders.parsers.pdf import (
    _format_inner_image,
    _purge_metadata,
)
from langchain_core.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import operations
else:
    UnstructuredClient = Any

Element: TypeAlias = Any

logger = logging.getLogger(__file__)

_DEFAULT_URL = "https://api.unstructuredapp.io/general/v0/general"

Type_extract_tables = Literal["text", "csv", "markdown", "html"]
Type_simple_mode = Literal["single", "elements"]
Type_all_mode = Literal["single", "elements", "paged", "page"]
Type_title = Literal["text", "markdown", "html"]
Type_image = Literal["text", "markdown-img", "html-img"]
Type_file_path = Union[str, PurePath, None]
# The historical version does not support tables or titles. It sometimes uses the
# default “elements” mode. It returns headers and footers, even though this disrupts
# text flow.
# The new version uses different default settings. It is still possible to force
# the settings to resume historical operation.
# The @deprecated declaration will mention the change of default values for parameters.
SIMULATES_THE_LEGACY_VERSION = False
if not SIMULATES_THE_LEGACY_VERSION:
    # Uses better default settings
    # The new approach favors the generation of a markdown flow.
    DEFAULT_MODE: Type_simple_mode = "single"
    DEFAULT_PAGE_DELIMITER = "\n\f\n"
    DEFAULT_KEEP_HEADER_FOOTER = False
    DEFAULT_EXTRACT_TABLES: Type_extract_tables = "markdown"
    DEFAULT_EXTRACT_TITLE: Type_title = "markdown"
    DEFAULT_IMAGE_FORMAT: Type_image = "markdown-img"
    DEFAULT_EPUB_MODE = DEFAULT_MODE
    DEFAULT_UNSTRUCTURED_MODE = DEFAULT_MODE
else:
    # To be compatiable with the legacy, use differents defaults values.
    # The test_migration.py file compares the behavior of the historical version
    # with this version, to ensure that the behavior is identical in all situations.
    # The original behavior is to retrieve a text stream, not to obtain a markdown.
    DEFAULT_MODE = "single"
    DEFAULT_PAGE_DELIMITER = "\n\n"
    DEFAULT_KEEP_HEADER_FOOTER = True
    DEFAULT_EXTRACT_TABLES = "text"
    DEFAULT_EXTRACT_TITLE = "text"
    DEFAULT_IMAGE_FORMAT = "text"
    DEFAULT_EPUB_MODE = "elements"
    DEFAULT_UNSTRUCTURED_MODE = "elements"


def _transform_cell_content(value: str, conversion_ind: bool = False) -> str:
    if value and conversion_ind is True:
        from markdownify import markdownify as md  # type: ignore[import-untyped]

        value = md(value)
    chars = {"|": "&#124;", "\n": "<br>"}
    for char, replacement in chars.items():
        value = value.replace(char, replacement)
    return value


def convert_table(
    html: str, content_conversion_ind: bool = False, all_cols_alignment: str = "left"
) -> str:
    alignment_options = {"left": " :--- ", "center": " :---: ", "right": " ---: "}
    if all_cols_alignment not in alignment_options.keys():
        raise ValueError(
            "Invalid alignment option for {!r} arg. Expected one of: {}".format(
                "all_cols_alignment", list(alignment_options.keys())
            )
        )

    soup = BeautifulSoup(html, "html.parser")

    if not soup.find():
        return html

    table = []
    table_body = []
    table_tr = list(soup.find_all("tr"))

    try:
        table_headings = [
            " "
            + _transform_cell_content(
                th.renderContents().decode("utf-8"),
                conversion_ind=content_conversion_ind,
            )
            + " "
            for th in soup.find("tr").find_all("th")  # type: ignore[union-attr]
        ]
    except AttributeError:
        raise ValueError("No {!r} tag found".format("tr"))

    if table_headings:
        table.append(table_headings)
        table_tr = table_tr[1:]

    for tr in table_tr:
        td_list = [
            " "
            + _transform_cell_content(
                td.renderContents().decode("utf-8"),  # type: ignore[union-attr]
                conversion_ind=content_conversion_ind,
            )
            + " "
            for td in list(tr.find_all("td"))  # type: ignore[attr-defined]
        ]
        table_body.append(td_list)

    table += table_body
    md_table_header = "|".join(
        [""]
        + ([" "] * len(table[0]) if not table_headings else table_headings)
        + ["\n"]
        + [alignment_options[all_cols_alignment]] * len(table[0])
        + ["\n"]
    )

    md_table = md_table_header + "".join(
        "|".join([""] + row + ["\n"]) for row in table_body
    )
    return md_table


def satisfies_min_unstructured_version(min_version: str) -> bool:
    """Check if the installed `Unstructured` version exceeds the minimum version
    for the feature in question."""
    from unstructured.__version__ import __version__ as __unstructured_version__

    min_version_tuple = tuple([int(x) for x in min_version.split(".")])

    # NOTE(MthwRobinson) - enables the loader to work when you're using pre-release
    # versions of unstructured like 0.4.17-dev1
    _unstructured_version = __unstructured_version__.split("-")[0]
    unstructured_version_tuple = tuple(
        [int(x) for x in _unstructured_version.split(".")]
    )

    return unstructured_version_tuple >= min_version_tuple


def satisfies_min_pdfminer_six_version(min_version: str) -> bool:
    """Check if the installed `Unstructured` version exceeds the minimum version
    for the feature in question."""
    from pdfminer import __version__ as __pdfminer_version__

    min_version_int = int(min_version)

    pdfminer_version = int(__pdfminer_version__)

    return pdfminer_version >= min_version_int


def validate_unstructured_version(min_unstructured_version: str) -> None:
    """Raise an error if the `Unstructured` version does not exceed the
    specified minimum."""
    if not satisfies_min_unstructured_version(min_unstructured_version):
        raise ValueError(
            f"unstructured>={min_unstructured_version} is required in this loader."
        )


class _UnstructuredDocumentParser(BaseBlobParser):
    """Provides loader functionality for individual document/file objects.

    Encapsulates partitioning individual blob objects either
    locally or via the Unstructured API.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        *,
        partition_via_api: bool = False,
        post_processors: Iterable[Callable[[str], str]] | None = None,
        mode: Type_all_mode,
        keep_header_footer: bool = DEFAULT_KEEP_HEADER_FOOTER,
        extract_tables: Type_extract_tables = DEFAULT_EXTRACT_TABLES,
        pages_delimitor: str = DEFAULT_PAGE_DELIMITER,
        images_parser: BaseImageBlobParser | None = None,
        images_inner_format: Type_image = DEFAULT_IMAGE_FORMAT,
        _title_format: Type_title = DEFAULT_EXTRACT_TITLE,
        # SDK parameters
        api_key: Optional[str] = None,
        client: Optional[UnstructuredClient] = None,
        url: Optional[str] = None,
        **unstructured_kwargs: Any,
    ):
        """unstructured base loader.

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
        if not satisfies_min_unstructured_version("0.17.2"):
            raise ValueError('unstructured>="0.17.2" is required in this loader.')
        _valid_modes = {"single", "elements", "paged", "page", None}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )
        _valid_extract_tables={"text", "csv", "markdown", "html"}
        if extract_tables not in _valid_extract_tables:
            raise ValueError(
                f"Got {extract_tables} for `extract_tables`, but should be one "
                f"of `{_valid_extract_tables}`"
            )

        if mode is None:
            mode = "elements"
        self._check_if_both_mode_and_chunking_strategy_are_by_page(
            mode, unstructured_kwargs
        )

        self.mode = mode
        if client is not None:
            disallowed_params = [("api_key", api_key), ("url", url)]
            bad_params = [
                param for param, value in disallowed_params if value is not None
            ]

            if bad_params:
                raise ValueError(
                    "if you are passing a custom `client`, you cannot also pass these "
                    f"params: {', '.join(bad_params)}."
                )
        self.client = client

        self.partition_via_api = partition_via_api
        self.post_processors = post_processors

        self.extract_tables = extract_tables
        self.title_format = _title_format
        self.pages_delimitor = pages_delimitor
        self.images_parser = images_parser
        self.images_inner_format = images_inner_format
        self.keep_header_footer = keep_header_footer
        if self.extract_tables != "text":
            unstructured_kwargs["infer_table_structure"] = True
        if self.mode != "elements":
            unstructured_kwargs["include_page_breaks"] = True

        # SDK parameters
        self.api_key = api_key or os.getenv("UNSTRUCTURED_API_KEY", "")
        self.url = url or os.getenv("UNSTRUCTURED_URL", "") or _DEFAULT_URL
        self.unstructured_kwargs = unstructured_kwargs

        if partition_via_api:
            try:
                from unstructured_client import UnstructuredClient

                self.client = client or UnstructuredClient(
                    api_key_auth=self.api_key, server_url=self.url
                )
            except ImportError:
                raise ImportError(
                    "unstructured package not found, please install it with "
                    "`pip install 'langchain_unstructured[client]'`"
                )

    def _check_if_both_mode_and_chunking_strategy_are_by_page(
        self, mode: str, unstructured_kwargs: dict[str, Any]
    ) -> None:
        if (
            mode in ("paged", "page")
            and unstructured_kwargs.get("chunking_strategy") == "by_page"
        ):
            raise ValueError(
                "Only one of `chunking_strategy='by_page'` or `mode='paged'` may be"
                " set. `chunking_strategy` is preferred."
            )

    def _elements_to_documents(
        self, file_metadata: dict[str, Any], elements_json: Iterable[dict]
    ) -> Iterator[Document]:
        for element in elements_json:
            metadata = file_metadata.copy()
            element_metadata = element.get("metadata") or {}
            metadata.update(element_metadata)
            metadata.update(
                {"category": element.get("category") or element.get("type")}
            )
            metadata.update({"element_id": element.get("element_id")})
            yield Document(
                id=element.get("element_id"),
                page_content=cast(str, element.get("text")),
                metadata=metadata,
            )

    def _convert_table(self, doc: Document) -> str:
        html_table = doc.metadata.get("text_as_html")
        if not html_table or self.extract_tables == "text":
            return doc.page_content
        if self.extract_tables == "html":
            return html_table
        elif self.extract_tables == "markdown":
            try:
                from markdownify import markdownify as md

                return md(html_table)
            except ImportError:
                raise ImportError(
                    "markdownify package not found, please install it with "
                    "`pip install markdownify`"
                )
        elif self.extract_tables == "csv":
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas package not found, please install it "
                    "with `pip install pandas`"
                )
            return pd.read_html(html_table)[0].to_csv()
        elif self.extract_tables == "text":
            return doc.page_content
        else:
            raise ValueError("extract_tables must be text, csv, markdown or html")

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        elements_json = (
            self._post_process_elements_json(self._get_elements_json(blob))
            if self.post_processors
            else self._get_elements_json(blob)
        )

        with blob.as_bytes_io() as file_obj:
            doc_metadata = _purge_metadata(
                self._get_metadata(file_obj, blob) | {"source": blob.source},
            )

        if self.mode == "elements":
            yield from self._yield_elements(doc_metadata, elements_json)
        elif self.mode in (
            "page",
            "paged",
            "single",
        ):
            page_contents: list[str] = []
            old_page_number = 1
            new_metadata: dict[str, Any] = {}
            page_number = 1
            for element in self._elements_to_documents(doc_metadata, elements_json):
                page_number = element.metadata.get("page_number", page_number)
                if (
                    element.metadata.get("category") == "PageBreak"
                    or page_number != old_page_number
                ):
                    if self.mode in ("page", "paged"):
                        if element.metadata.get("category") == "PageBreak":
                            if element.page_content:
                                page_contents.append(element.page_content)
                        new_metadata.pop("category", "")
                        yield self._yield_page(
                            id=element.id,
                            page_contents=page_contents,
                            doc_metadata=doc_metadata,
                            elem_metadata=new_metadata,
                            page_number=page_number,
                        )
                        new_metadata.clear()
                        page_contents.clear()
                        if element.metadata.get("category") == "PageBreak":
                            page_number += 1
                    else:
                        if element.metadata.get("category") == "PageBreak":
                            if element.page_content:
                                page_contents.append(element.page_content)
                            page_contents.append(self.pages_delimitor)
                            page_number += 1

                if element.metadata.get("category") == "PageBreak":
                    pass
                elif element.metadata.get("category") == "Image":
                    if self.images_parser:
                        if "image_path" in element.metadata:
                            blob = Blob.from_path(element.metadata["image_path"])
                            image_text = next(
                                self.images_parser.lazy_parse(blob)
                            ).page_content
                            if image_text:
                                page_contents.append(
                                    _format_inner_image(
                                        blob, image_text, self.images_inner_format
                                    )
                                    + "\n\n"
                                )
                    else:
                        page_contents.append(element.page_content + "\n\n")
                    new_metadata |= element.metadata
                elif element.metadata.get("category") == "Table":
                    page_contents.append(
                        self._convert_table(
                            element,
                        )
                        + "\n\n"
                    )
                    new_metadata |= element.metadata
                elif element.metadata.get("category") == "Title":
                    page_contents.append(
                        self._convert_title(element.page_content) + "\n\n"
                    )
                    new_metadata |= element.metadata
                elif element.metadata.get("category") == "Header":
                    if self.keep_header_footer:
                        page_contents.append(
                            self._convert_title(element.page_content) + "\n\n"
                        )
                        new_metadata |= element.metadata
                elif element.metadata.get("category") == "Footer":
                    if self.keep_header_footer:
                        page_contents.append(
                            self._convert_title(element.page_content) + "\n\n"
                        )
                        new_metadata |= element.metadata
                else:
                    # NarrativeText, UncategorizedText, Formula, FigureCaption,
                    # ListItem, Address, EmailAddress
                    if element.metadata.get("category") not in [
                        "NarrativeText",
                        "UncategorizedText",
                        "Formula",
                        "FigureCaption",
                        "ListItem",
                        "Address",
                        "EmailAddress",
                    ]:
                        logger.warning(
                            "Unknown category %s", element.metadata.get("category")
                        )
                    page_contents.append(element.page_content + "\n\n")
                    new_metadata |= element.metadata

                old_page_number = page_number
            if self.mode == "single":
                new_metadata.pop("category", "")
                yield self._yield_single(
                    page_contents=page_contents,
                    doc_metadata=doc_metadata,
                    elem_metadata=new_metadata,
                )
            else:
                if page_contents:
                    new_metadata.pop("category", "")
                    yield self._yield_page(
                        id=None,
                        page_contents=page_contents,
                        doc_metadata=doc_metadata,
                        elem_metadata=new_metadata,
                        page_number=page_number,
                    )
                pass

    def _convert_title(self, text: str) -> str:
        if self.title_format == "markdown":
            return "# " + text
        elif self.title_format == "html":
            return "<h1>" + text + "</h1>"
        return text

    def _yield_page(
        self,
        *,
        id: str | None,
        page_contents: List[str],
        doc_metadata: dict[str, Any],
        elem_metadata: dict[str, Any],
        page_number: int,
    ) -> Document:
        return Document(
            id=id,
            page_content="".join(page_contents),
            metadata=doc_metadata
            | elem_metadata
            | {"category": "CompositeElement", "page": page_number},
        )

    def _yield_single(
        self,
        page_contents: List[str],
        doc_metadata: dict[str, Any],
        elem_metadata: dict[str, Any],
    ) -> Document:
        # Ignore elem_metadata, but not in subclasses
        return Document(
            page_content="".join(page_contents).strip(),
            metadata=doc_metadata
            | elem_metadata
            | {
                "category": "CompositeElement",
            },
        )

    def _yield_elements(
        self, doc_metadata: dict[str, Any], elements_json: Iterable[dict[str, Any]]
    ) -> Iterator[Document]:
        for element in elements_json:
            metadata = deepcopy(doc_metadata)
            metadata.update(cast(dict[str, Any], element.get("metadata")))
            metadata.update(
                {"category": element.get("category") or element.get("type")}
            )
            metadata.update({"element_id": element.get("element_id")})
        yield from self._elements_to_documents(doc_metadata, elements_json)

    def _get_elements_json(self, blob: Blob) -> list[dict[str, Any]]:
        """Get elements as a list of dictionaries from local partition or via API."""
        if self.partition_via_api:
            return self._get_elements_via_api(blob)

        return self._convert_elements_to_dicts(self._get_elements_via_local(blob))

    @abstractmethod
    def _get_elements_via_local(self, blob: Blob) -> list[Element]: ...

    def _get_elements_via_api(self, blob: Blob) -> list[dict[str, Any]]:
        """Retrieve a list of element dicts from the API using the SDK client."""
        if not self.client:
            try:
                from unstructured_client import UnstructuredClient

                self.client = UnstructuredClient(
                    api_key_auth=self.api_key,
                    server_url=self.url or _DEFAULT_URL,
                )

            except ImportError:
                raise ImportError(
                    "unstructured package not found, please install it with "
                    "`pip install 'langchain_unstructured[client]'`"
                )

        req = self._sdk_partition_request(blob)
        response = self.client.general.partition(request=req)
        if response.status_code == 200:
            return json.loads(response.raw_response.text)
        raise ValueError(
            f"Receive unexpected status code {response.status_code} from the API.",
        )

    def _sdk_partition_request(self, blob: Blob) -> operations.PartitionRequest:
        from unstructured_client.models import (
            operations,
            shared,
        )

        return operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=blob.as_bytes(),
                    file_name=str(blob.path),
                    content_type=blob.mimetype,
                ),
                **self.unstructured_kwargs,
            ),
        )

    def _convert_elements_to_dicts(
        self, elements: list[Element]
    ) -> list[dict[str, Any]]:
        return [element.to_dict() for element in elements]

    def _get_metadata(
        self,
        fp: BinaryIO,
        blob: Blob,
    ) -> dict[str, Any]:
        return blob.metadata

    def _post_process_elements_json(
        self, elements_json: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        if self.post_processors:
            for element in elements_json:
                for post_processor in self.post_processors:
                    element["text"] = post_processor(str(element.get("text")))
        return elements_json
