"""Loader that uses unstructured to load HTML files."""

import logging
from typing import Any, Iterator, List, Literal, cast

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_unstructured.parsers.base import DEFAULT_MODE

logger = logging.getLogger(__name__)


class UnstructuredURLLoader(BaseLoader):
    """Load files from remote URLs using `Unstructured`.

    Use the unstructured partition function to detect the MIME type
    and route the file to the appropriate partitioner.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredURLLoader

    loader = UnstructuredURLLoader(
        urls=["<url-1>", "<url-2>"], mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition
    """

    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        mode: Literal["single", "elements"] = DEFAULT_MODE,
        show_progress_bar: bool = False,
        **unstructured_kwargs: Any,
    ):
        """Initializes the UnstructuredURLLoader.

        Args:
            urls: A list of URLs to load documents from.
            continue_on_failure: If True, the loader will continue processing
                other URLs if one fails. If False, it will raise an exception
                on the first failure. Defaults to True.
            mode: The mode for partitioning the document. Options are:
                - "single": Returns the entire document content as a single Document.
                - "elements": Returns each identified element (e.g., Title,
                  NarrativeText) as a separate Document.
                Defaults to the value of `DEFAULT_MODE` (typically "single").
            show_progress_bar: If True, displays a progress bar using `tqdm`
                while processing the URLs. Requires `tqdm` to be installed.
                Defaults to False.
            **unstructured_kwargs: Additional keyword arguments to pass directly
                to the underlying `unstructured` partitioning function
                (specifically `partition_html` for URLs). Common arguments
                include `strategy`, `encoding`, and `headers`. Note that
                `headers` are extracted and handled separately by this loader,
                and their applicability might depend on the installed
                `unstructured` version.
        """
        try:
            import unstructured  # noqa:F401
            from unstructured.__version__ import __version__ as __unstructured_version__

            self.__version = __unstructured_version__
        except ImportError:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        self._validate_mode(mode)
        self.mode = mode

        headers = unstructured_kwargs.pop("headers", {})
        if len(headers.keys()) != 0:
            warn_about_headers = False
            if self.__is_non_html_available():
                warn_about_headers = not self.__is_headers_available_for_non_html()
            else:
                warn_about_headers = not self.__is_headers_available_for_html()

            if warn_about_headers:
                logger.warning(
                    "You are using an old version of unstructured. "
                    "The headers parameter is ignored"
                )

        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headers = headers
        self.unstructured_kwargs = unstructured_kwargs
        self.show_progress_bar = show_progress_bar

    def __is_headers_available_for_html(self) -> bool:
        _unstructured_version = self.__version.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        return unstructured_version >= (0, 5, 7)

    def __is_headers_available_for_non_html(self) -> bool:
        _unstructured_version = self.__version.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        return unstructured_version >= (0, 5, 13)

    def __is_non_html_available(self) -> bool:
        _unstructured_version = self.__version.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        return unstructured_version >= (0, 5, 12)

    def _validate_mode(self, mode: str) -> None:
        _valid_modes = {"single", "elements"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        try:
            from unstructured.partition.html import partition_html
        except ImportError as e:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install 'langchain_unstructured[all_docs]'`"
            ) from e

        urls: list[str]
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
            except ImportError as e:
                raise ImportError(
                    "Package tqdm must be installed if show_progress_bar=True. "
                    "Please install with 'pip install tqdm' or set "
                    "show_progress_bar=False."
                ) from e

            urls = cast(list[str], tqdm(self.urls))
        else:
            urls = self.urls

        for url in urls:
            try:
                elements = partition_html(
                    url=url, headers=self.headers, **self.unstructured_kwargs
                )
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                    continue
                else:
                    raise e

            if self.mode == "single":
                text = "\n\n".join([str(el) for el in elements])
                metadata = {"source": url}
                yield Document(page_content=text, metadata=metadata)
            elif self.mode == "elements":
                for element in elements:
                    metadata = element.metadata.to_dict()
                    metadata["category"] = element.category
                    yield Document(page_content=str(element), metadata=metadata)
