from __future__ import annotations

import mimetypes
import io
from html.parser import HTMLParser
from typing import List, Optional

from langchain.schema import Document
from pypdf import PdfReader


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _guess_type(filename: str) -> str:
    ext = (filename or "").lower()
    if ext.endswith(".md"):
        return "text/markdown"
    if ext.endswith(".txt"):
        return "text/plain"
    if ext.endswith(".pdf"):
        return "application/pdf"
    if ext.endswith(".html") or ext.endswith(".htm"):
        return "text/html"
    mt, _ = mimetypes.guess_type(filename or "")
    return mt or "application/octet-stream"


def load_document_from_bytes(
    filename: str,
    data: bytes,
    source: Optional[str] = None,
) -> List[Document]:
    file_type = _guess_type(filename)
    source_value = source or filename

    if file_type == "application/pdf":
        reader = PdfReader(io.BytesIO(data or b""))
        text_parts: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                text_parts.append(t)
        text = "\n\n".join(text_parts)
        return [Document(page_content=text, metadata={"source": source_value, "file_type": "pdf"})]

    raw = (data or b"").decode("utf-8", errors="ignore")
    if file_type == "text/html":
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        text = parser.get_text()
        return [Document(page_content=text, metadata={"source": source_value, "file_type": "html"})]

    if file_type in {"text/plain", "text/markdown"}:
        return [
            Document(
                page_content=raw,
                metadata={"source": source_value, "file_type": "md" if file_type == "text/markdown" else "txt"},
            )
        ]

    return [Document(page_content=raw, metadata={"source": source_value, "file_type": "unknown"})]
