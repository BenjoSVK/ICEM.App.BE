from pathlib import Path
import zipfile
import logging
from typing import List
from schemas.file_info import FileInfo

from backend.storage import Storage
from backend.file_handlers import IFileHandler, IFileHandlers


logger = logging.getLogger("uvicorn.access")


class ZipArchiveHandler(IFileHandler):
    EXTENSIONS = [
        ".zip"
    ]

    def is_supported(self, file_path: Path) -> bool:
        ext = file_path.suffix.lower()
        return ext in ZipArchiveHandler.EXTENSIONS
    

    def accept_file(
        self, 
        file_path: Path,
        storage: Storage,
        handlers: IFileHandlers
    ) -> List[FileInfo]:
        """Triggers file pre-processing on the uploaded (ZIP) file.
        Internal content should be extracted, and handled according 
        to the detected file format.

        Returns:
            List[FileInfo] - contains the list of files extracted
            from the given input archive.

        Method throws exceptions on error.            
        """        
        result = []

        logger.info(f"ZipArchiveHandler.accept_file - {file_path.as_posix()}")

        if not file_path.exists():
            logger.error(f"Input archive file '{file_path}' not found.")
            return []

        
        # Extract the files to a local temp folder
        with storage.get_temp_folder() as temp_dir:
            
            # Extract the archive
            with zipfile.ZipFile(file_path, "r") as archive:
                archive.extractall(temp_dir.as_posix())

            # Now loop over all the files, and try and accept them
            extracted_files = sorted(list(temp_dir.rglob("*")))
            for inner_file in extracted_files:
                try:
                    inner_result = handlers.accept_file(inner_file, storage)

                    # Append the list of results
                    result += inner_result
                except:
                    pass
                

        return result

