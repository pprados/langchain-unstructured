"""Loads Microsoft Excel files."""

from typing import IO, TYPE_CHECKING, Any, Callable, Iterable, Optional

from langchain_community.document_loaders.parsers.images import BaseImageBlobParser
from langchain_community.document_loaders.unstructured import (
    validate_unstructured_version,
)

from langchain_unstructured.base import UnstructuredClient
from langchain_unstructured.parsers.base import (
    DEFAULT_MODE,
    Type_file_path,
    Type_simple_mode,
)
from langchain_unstructured.parsers.image import UnstructuredImageParser
from langchain_unstructured.unstructured import _UnstructuredBaseLoader

if TYPE_CHECKING:
    from PIL.Image import Image


class UnstructuredImageLoader(_UnstructuredBaseLoader, BaseImageBlobParser):
    """Load `PNG` and `JPG` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Setup:
        Install ``langchain-unstructured[image]``.

        .. code-block:: bash

            pip install -U langchain-unstructured[image]

    Instantiate:
        .. code-block:: python

            from langchain_unstructured import UnstructuredImageLoader

            loader = UnstructuredImageLoader(
                "example.png",
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
        file_path: Type_file_path = None,
        *,
        file: Optional[IO[bytes]] = None,
        partition_via_api: bool = False,
        post_processors: Optional[Iterable[Callable[[str], str]]] = None,
        mode: Type_simple_mode = DEFAULT_MODE,
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
                - "elements": Returns each identified element (e.g., title, text,
                  table) as a separate Document.
                Defaults to "single".
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
        super().__init__(
            parser=UnstructuredImageParser(
                partition_via_api=partition_via_api,
                post_processors=post_processors,
                mode=mode,
                api_key=api_key,
                client=client,
                url=url,
                **unstructured_kwargs,
            ),
            file_path=file_path,
            file=file,
            web_url=web_url,
        )

    def _analyze_image(self, img: "Image") -> str:
        return ""
