from pathlib import Path
from typing import Any, Tuple, Type

import pytest
from langchain_community.document_loaders import (
    UnstructuredCSVLoader as old_UnstructuredCSVLoader,
)
from langchain_community.document_loaders.email import (
    UnstructuredEmailLoader as old_UnstructuredEmailLoader,
)
from langchain_community.document_loaders.epub import (
    UnstructuredEPubLoader as old_UnstructuredEPubLoader,
)
from langchain_community.document_loaders.excel import (
    UnstructuredExcelLoader as old_UnstructuredExcelLoader,
)
from langchain_community.document_loaders.html import (
    UnstructuredHTMLLoader as old_UnstructuredHTMLLoader,
)
from langchain_community.document_loaders.image import (
    UnstructuredImageLoader as old_UnstructuredImageLoader,
)
from langchain_community.document_loaders.markdown import (
    UnstructuredMarkdownLoader as old_UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.odt import (
    UnstructuredODTLoader as old_UnstructuredODTLoader,
)
from langchain_community.document_loaders.org_mode import (
    UnstructuredOrgModeLoader as old_UnstructuredOrgModeLoader,
)
from langchain_community.document_loaders.pdf import (
    UnstructuredPDFLoader as old_UnstructuredPDFLoader,
)
from langchain_community.document_loaders.powerpoint import (
    UnstructuredPowerPointLoader as old_UnstructuredPowerPointLoader,
)
from langchain_community.document_loaders.rst import (
    UnstructuredRSTLoader as old_UnstructuredRSTLoader,
)
from langchain_community.document_loaders.rtf import (
    UnstructuredRTFLoader as old_UnstructuredRTFLoader,
)
from langchain_community.document_loaders.tsv import (
    UnstructuredTSVLoader as old_UnstructuredTSVLoader,
)
from langchain_community.document_loaders.url import (
    UnstructuredURLLoader as old_UnstructuredURLLoader,
)
from langchain_community.document_loaders.word_document import (
    UnstructuredWordDocumentLoader as old_UnstructuredWordDocumentLoader,
)
from langchain_community.document_loaders.xml import (
    UnstructuredXMLLoader as old_UnstructuredXMLLoader,
)

from langchain_unstructured import (
    UnstructuredCSVLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredOrgModeLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredRTFLoader,
    UnstructuredTSVLoader,
    UnstructuredURLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredXMLLoader,
)
from langchain_unstructured.parsers.base import (
    SIMULATES_THE_LEGACY_VERSION,
    satisfies_min_unstructured_version,
)
from langchain_unstructured.unstructured import UnstructuredLoader

# from langchain_community.document_loaders.unstructured import (
#     UnstructuredFileLoader as old_UnstructuredLoader,
# )
from .document_loaders_old import (
    UnstructuredLoader as old_UnstructuredLoader,
)

EXAMPLE_DOCS_DIRECTORY = Path(__file__).parent.parent / "examples"


def _test_migration(classes: Tuple[Type, Type], file_path: Path, kwargs: Any) -> None:
    """Test unstructured loader."""
    class_old, class_new = classes
    loader_old = class_old(str(file_path), **kwargs)
    loader_new = class_new(
        file_path,
        # To be similare to the previous version
        # set these parameters
        _title_format=None,  # It's a private parameter
        **kwargs,
    )
    result_old = loader_old.load()
    result_new = loader_new.load()

    assert not hasattr(loader_old, "mode") or loader_old.mode == loader_new.parser.mode
    assert len(result_old) == len(result_new)
    for doc_old, doc_new in zip(result_old, result_new):
        assert doc_new.page_content.strip() == doc_old.page_content.strip()
        for k, v in doc_old.metadata.items():
            assert v == doc_new.metadata[k]


def append_the_end(text: str) -> str:
    return text + " THE END!"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "extract_tables": "text", "post_processors": [append_the_end]},
        {
            "mode": "elements",
            "extract_tables": "text",
            "post_processors": [append_the_end],
        },
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_csv_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredCSVLoader, UnstructuredCSVLoader),
        EXAMPLE_DOCS_DIRECTORY / "stanley-cups.csv",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
        {"mode": "single", "process_attachments": True},
        {"mode": "elements", "process_attachments": True},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_email_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredEmailLoader, UnstructuredEmailLoader),
        EXAMPLE_DOCS_DIRECTORY / "fake-email-attachment.eml",
        kwargs,
    )
    _test_migration(
        (old_UnstructuredEmailLoader, UnstructuredEmailLoader),
        EXAMPLE_DOCS_DIRECTORY / "hello.msg",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_epub_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredEPubLoader, UnstructuredEPubLoader),
        EXAMPLE_DOCS_DIRECTORY / "epub30-spec.epub",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "paged", "post_processors": None},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_excel_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredExcelLoader, UnstructuredExcelLoader),
        EXAMPLE_DOCS_DIRECTORY / "stanley-cups.xlsx",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
        {"mode": "single", "extract_tables": "html"},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_html_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredHTMLLoader, UnstructuredHTMLLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.html",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_image_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredImageLoader, UnstructuredImageLoader),
        EXAMPLE_DOCS_DIRECTORY / "image.png",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_markdown_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredMarkdownLoader, UnstructuredMarkdownLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.md",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_odt_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredODTLoader, UnstructuredODTLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.odt",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
        {"mode": "single", "extract_table": "html"},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_org_mode_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredOrgModeLoader, UnstructuredOrgModeLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.org",
        kwargs,
    )


_default_args = {
    "pages_delimitor": "\n\n",
    "include_page_breaks": True,
    "extract_tables": "text",
}


@pytest.mark.parametrize(
    "kwargs",
    [
        {} | _default_args,
        {"mode": "elements", "post_processors": [append_the_end]},
        {"mode": "single", "post_processors": [append_the_end]} | _default_args,
        {"mode": "paged", "post_processors": None} | _default_args,
        {"mode": "single", "strategy": "auto"} | _default_args,
        {"mode": "single", "strategy": "fast"} | _default_args,
        {"mode": "single", "strategy": "hi_res", "keep_header_footer": True}
        | _default_args,
        {"mode": "single", "strategy": "ocr_only"} | _default_args,
        {"mode": "paged", "strategy": "auto"} | _default_args,
        {"mode": "paged", "strategy": "fast"} | _default_args,
        {"mode": "paged", "strategy": "hi_res", "keep_header_footer": True}
        | _default_args,
        {"mode": "paged", "strategy": "ocr_only"} | _default_args,
    ],
)
@pytest.mark.local
@pytest.mark.skipif(
    not satisfies_min_unstructured_version("0.17.3"),
    reason="The result is random. See pr 3978 in unstructured",
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
def test_migration_unstructured_pdf_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredPDFLoader, UnstructuredPDFLoader),
        EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "paged", "post_processors": None},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_powerpoint_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredPowerPointLoader, UnstructuredPowerPointLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.pptx",
        kwargs,
    )
    _test_migration(
        (old_UnstructuredPowerPointLoader, UnstructuredPowerPointLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.ppt",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_rst_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredRSTLoader, UnstructuredRSTLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.rst",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_rtf_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredRTFLoader, UnstructuredRTFLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.rtf",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_tsv_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredTSVLoader, UnstructuredTSVLoader),
        EXAMPLE_DOCS_DIRECTORY / "stanley-cups.tsv",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredLoader, UnstructuredLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.html",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_url_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    classes = (old_UnstructuredURLLoader, UnstructuredURLLoader)
    urls = [
        "https://docs.unstructured.io/",
        "http://www.google.com/",
    ]
    class_old, class_new = classes
    result_old = class_old(urls=urls, **kwargs).load()
    result_new = class_new(urls=urls, **kwargs).load()

    assert len(result_old) == len(result_new)
    for doc_old, doc_new in zip(result_old, result_new):
        assert doc_old.page_content.strip() == doc_new.page_content.strip()
        for k, v in doc_old.metadata.items():
            if k == "link_urls":
                continue
            assert doc_new.metadata[k] == v, f"key {k}: {doc_new.metadata[k]} != {v}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "paged", "post_processors": None},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_word_document_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredWordDocumentLoader, UnstructuredWordDocumentLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.docx",
        kwargs,
    )
    _test_migration(
        (old_UnstructuredWordDocumentLoader, UnstructuredWordDocumentLoader),
        EXAMPLE_DOCS_DIRECTORY / "README.doc",
        kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "single", "post_processors": [append_the_end]},
        {"mode": "elements", "post_processors": [append_the_end]},
    ],
)
@pytest.mark.skipif(
    not SIMULATES_THE_LEGACY_VERSION,
    reason="The parameters do not allow comparison with the historical version",
)
@pytest.mark.local
def test_migration_unstructured_xml_loader(kwargs: Any) -> None:
    """Test unstructured loader."""
    _test_migration(
        (old_UnstructuredXMLLoader, UnstructuredXMLLoader),
        EXAMPLE_DOCS_DIRECTORY / "factbook.xml",
        kwargs,
    )
