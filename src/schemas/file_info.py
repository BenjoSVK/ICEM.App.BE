

from pydantic import BaseModel
from datetime import datetime


class FileInfo(BaseModel):
    id: str
    last_modified: str
    size_bytes: int

