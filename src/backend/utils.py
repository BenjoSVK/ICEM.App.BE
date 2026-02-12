"""
Utilities for building FileInfo from filesystem paths.
"""
from datetime import datetime
from pathlib import Path

from schemas.file_info import FileInfo


def get_file_info(file_path: Path, base_path: Path) -> FileInfo:
    """Build FileInfo (id, last_modified, size_bytes) for a file relative to base_path."""
    last_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)    
    return FileInfo(
        id=file_path.relative_to(base_path).as_posix(),
        last_modified=last_modified_time.strftime("%Y-%m-%d"),
        size_bytes=file_path.stat().st_size
    )