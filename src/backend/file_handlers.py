
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
from schemas.file_info import FileInfo
from backend.storage import Storage



class IFileHandlers(ABC):
    @abstractmethod
    def accept_file(self, file_path: Path, storage: Storage) -> List[FileInfo]:
        pass


class IFileHandler(ABC):
    @abstractmethod
    def is_supported(self, file_path: Path) -> bool:
        pass

    @abstractmethod
    def accept_file(
        self, 
        file_path: Path,
        storage: Storage,
        handlers: IFileHandlers
    ) -> List[FileInfo]:
        pass





class FileHandlers(IFileHandlers):
    def __init__(
        self,
        handlers: List[IFileHandler]
    ):
        self.handlers = handlers


    def accept_file(self, file_path: Path, storage: Storage) -> List[FileInfo]:
        """We loop over all our handlers and try to
        use them all to extract/process all files. If image
        files are contained within an archive, they will all be
        repeatedly processed with all our handlers.
        """
        
        result = []

        handler = self._get_handler(file_path)
        if not handler:
            # TODO: log !
            #raise ValueError(f"No suitable handler found for file: {file_path.name}")
            return []

        # Delegate        
        result = handler.accept_file(file_path, storage, self)

        # Return what we've got
        return result
    


    def _get_handler(self, file_path: Path) -> IFileHandler:
        for handler in self.handlers:
            if handler.is_supported(file_path):
                return handler
            
        return None
    
