"""
Pydantic models for file metadata (id, last_modified, size_bytes).
"""
from pydantic import BaseModel


class FileInfo(BaseModel):
    """Metadata for a single file (TIFF or result) listed by the API."""

    id: str
    last_modified: str
    size_bytes: int

