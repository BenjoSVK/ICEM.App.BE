"""
Handler for image files: copies supported TIFF/WSI files into the TIFF folder.
"""
import logging
import shutil
from pathlib import Path
from typing import List

from schemas.file_info import FileInfo

from backend.file_handlers import IFileHandler, IFileHandlers
from backend.storage import Storage
from backend.utils import get_file_info

logger = logging.getLogger("uvicorn.access")


class ImageFileHandler(IFileHandler):
    """Accepts .tif, .tiff, .wsi, .vsi files and copies them into the TIFF folder."""

    EXTENSIONS = [".tif", ".tiff", ".wsi", ".vsi"]

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

        logger.info(f"ImageFileHandler.accept_file - {file_path.as_posix()}")

        if not file_path.exists():
            logger.error(f"Input file '{file_path}' not found.")
            return []
        
        # Target file path
        tif_folder = storage.get_folderpath(Storage.TIF_FOLDER)
        target = storage.get_filepath(Storage.TIF_FOLDER, file_path.name, True)

        # Copy the file
        shutil.copyfile(src=file_path, dst=target)

        # Return the info
        result.append(get_file_info(target, tif_folder))
        return result
