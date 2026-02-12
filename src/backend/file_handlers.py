"""
Abstract file handlers and composite that delegates to the first matching handler.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from schemas.file_info import FileInfo

from backend.storage import Storage

logger = logging.getLogger(__name__)


class IFileHandlers(ABC):
    """Interface for a composite that accepts a file and returns extracted FileInfo list."""

    @abstractmethod
    def accept_file(self, file_path: Path, storage: Storage) -> List[FileInfo]:
        pass


class IFileHandler(ABC):
    """Interface for a single handler that supports specific file types."""

    @abstractmethod
    def is_supported(self, file_path: Path) -> bool:
        pass

    @abstractmethod
    def accept_file(
        self,
        file_path: Path,
        storage: Storage,
        handlers: IFileHandlers,
    ) -> List[FileInfo]:
        pass





class FileHandlers(IFileHandlers):
    """Composite that delegates to the first handler that supports the file."""

    def __init__(self, handlers: List[IFileHandler]) -> None:
        self.handlers = handlers

    def accept_file(self, file_path: Path, storage: Storage) -> List[FileInfo]:
        """Use the first supporting handler to process the file; return extracted FileInfo list."""
        handler = self._get_handler(file_path)
        if not handler:
            logger.warning("No suitable handler found for file: %s", file_path.name)
            return []
        return handler.accept_file(file_path, storage, self)

    def _get_handler(self, file_path: Path) -> Optional[IFileHandler]:
        """Return the first handler that supports this file path, or None."""
        for handler in self.handlers:
            if handler.is_supported(file_path):
                return handler
        return None
    
