import os
from pathlib import Path
from typing import Callable, List

import pytest
from langchain_core.documents import Document

from langchain_unstructured.parsers.base import DEFAULT_UNSTRUCTURED_MODE
from langchain_unstructured.unstructured import UnstructuredLoader

EXAMPLE_DOCS_DIRECTORY = Path(__file__).parent.parent / "examples"


def _check_docs_content(docs: List[Document]) -> None:
    assert all(
        doc.metadata.get("filename") == "layout-parser-paper.pdf" for doc in docs
    )

    assert (
        docs[0].metadata.get("category") == "CompositeElement"
        or sum(doc.metadata.get("category") == "PageBreak" for doc in docs) == 16
    )  # 16 page doc

    expected_metadata_keys = [
        "source",
        "languages",
        "page_number",
        "category",
        "element_id",
    ]
    if docs[0].metadata.get("category") != "CompositeElement":
        expected_metadata_keys.append("coordinates")
        categories = set(doc.metadata.get("category") for doc in docs)
        assert "NarrativeText" in categories
        assert "Title" in categories

    for doc in docs:
        if doc.page_content:
            for key in expected_metadata_keys:
                assert key in doc.metadata
        else:
            assert doc.metadata.get("category") == "PageBreak"

    if len(docs) > 1:
        page_numbers = [
            doc.metadata.get("page_number")
            for doc in docs
            if doc.metadata.get("page_number")
        ]

        assert set(page_numbers) == set(range(1, 17))
        assert len(docs) >= 16  # (16 pages * (>=1 element per page) + 16 page breaks)

    if DEFAULT_UNSTRUCTURED_MODE != "elements":
        assert (
            "LayoutParser: A Uniﬁed Toolkit for Deep Learning "
            "Based Document Image Analysis"
        ) in docs[0].page_content


@pytest.mark.local
def test_loader_partitions_locally() -> None:
    file_path = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"

    docs = UnstructuredLoader(
        file_path=file_path,
        # Unstructured kwargs
        chunking_strategy="by_title",
        mode="page",
        strategy="fast",
        include_page_breaks=True,
    ).load()

    _check_docs_content(docs)


@pytest.mark.local
async def test_loader_partitions_locally_async_lazy() -> None:
    file_path = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"

    loader = UnstructuredLoader(
        file_path=file_path,
        # Unstructured kwargs
        chunking_strategy="by_title",
        strategy="fast",
        include_page_breaks=True,
    )
    docs = [doc async for doc in loader.alazy_load()]

    _check_docs_content(docs)


@pytest.mark.local
async def test_loader_partitions_locally_async_lazy_with_list() -> None:
    file_path = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"

    loader = UnstructuredLoader(
        file_path=[file_path, file_path],
        # Unstructured kwargs
        chunking_strategy="by_title",
        strategy="fast",
        include_page_breaks=True,
    )
    docs = [doc async for doc in loader.alazy_load()]

    _check_docs_content([docs[0]])
    _check_docs_content([docs[1]])


@pytest.mark.local
def test_loader_partitions_locally_and_applies_post_processors(
    get_post_processor: Callable[[str], str],
) -> None:
    file_path = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"
    loader = UnstructuredLoader(
        file_path=file_path,
        post_processors=[get_post_processor],
        strategy="fast",
    )

    docs = loader.load()

    if DEFAULT_UNSTRUCTURED_MODE == "single":
        assert len(docs) == 1
    else:
        assert len(docs) > 1
    assert docs[0].page_content.endswith("THE END!")


@pytest.mark.local
def test_unstructured_with_web_url_loader() -> None:
    docs = UnstructuredLoader(web_url="https://www.example.com/").load()

    for doc in docs:
        assert doc.page_content
        assert doc.metadata["filetype"] == "text/html"
        assert doc.metadata["url"] == "https://www.example.com/"
        assert doc.metadata["category"]


# -- API partition --


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
def test_loader_partitions_via_api() -> None:
    file_path = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"
    loader = UnstructuredLoader(
        file_path=file_path,
        partition_via_api=True,
        # Unstructured kwargs
        strategy="fast",
        include_page_breaks=True,
        coordinates=True,
    )

    docs = loader.load()

    _check_docs_content(docs)


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
async def test_loader_partitions_via_api_async_lazy() -> None:
    file_path = EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf"
    loader = UnstructuredLoader(
        file_path=file_path,
        partition_via_api=True,
        # Unstructured kwargs
        strategy="fast",
        include_page_breaks=True,
        coordinates=True,
    )

    docs = [doc async for doc in loader.alazy_load()]

    _check_docs_content(docs)


@pytest.mark.skipif(
    not os.environ.get("UNSTRUCTURED_API_KEY"), reason="Unstructured API key not found"
)
@pytest.mark.skip(
    reason="I don't currently implement path list management. "
    "I think GenericLoader is there for that."
)
def test_loader_partitions_multiple_via_api() -> None:
    file_paths = [
        EXAMPLE_DOCS_DIRECTORY / "layout-parser-paper.pdf",
        EXAMPLE_DOCS_DIRECTORY / "fake-email-attachment.eml",
    ]
    loader = UnstructuredLoader(
        file_path=file_paths,
        partition_via_api=True,
        # Unstructured kwargs
        strategy="fast",
    )

    docs = loader.load()

    assert len(docs) > 1
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"
    assert docs[-1].metadata.get("filename") == "fake-email-attachment.eml"


# -- fixtures ---


@pytest.fixture()
def get_post_processor() -> Callable[[str], str]:
    def append_the_end(text: str) -> str:
        return text + "THE END!"

    return append_the_end
