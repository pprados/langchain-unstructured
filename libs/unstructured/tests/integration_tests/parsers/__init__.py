from pathlib import Path
from typing import Tuple

from langchain_core.documents.base import Blob


def couple_of_blob(file_path: Path, mime_type: str) -> Tuple[Blob, Blob]:
    blob1 = Blob(path=file_path, mimetype=mime_type)
    with open(file_path, "rb") as f:
        data = f.read()
    blob2 = Blob.from_data(data, mime_type=mime_type)
    return blob1, blob2
