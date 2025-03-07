from pathlib import Path
import shutil
from typing import List
from schemas.file_info import FileInfo
from backend.utils import get_file_info

from backend.storage import Storage
from backend.file_handlers import IFileHandler, IFileHandlers



class ImageFileHandler(IFileHandler):
    EXTENSIONS = [
        ".tif", ".tiff", ".wsi", ".vsi"
    ]

    def is_supported(self, file_path: Path) -> bool:
        ext = file_path.suffix.lower()
        return ext in ImageFileHandler.EXTENSIONS


    def accept_file(
        self, 
        file_path: Path,
        storage: Storage,
        handlers: IFileHandlers
    ) -> List[FileInfo]:
        """Simply copy the file into the TIF_FOLDER folder."""

        result = []

        if not file_path.exists():
            raise FileNotFoundError(f"Input archive file '{file_path}' not found.")
        
        # Target file path
        tif_folder = storage.get_folderpath(Storage.TIF_FOLDER)
        target = storage.get_filepath(Storage.TIF_FOLDER, file_path.name, True)

        # Copy the file
        shutil.copyfile(src=file_path, dst=target)

        # Return the info
        print(f"  - Accepted file: {target}")
        result.append(get_file_info(target, tif_folder))
        return result
