"""
Inference backend: accepts uploads, lists TIFF/result files, and runs inference via an engine.
"""
import logging
from pathlib import Path
from typing import List, Optional

from backend.file_handlers import FileHandlers
from backend.storage import Storage
from backend.utils import get_file_info
from inference.inference_engine import InferenceEngine
from schemas.file_info import FileInfo

logger = logging.getLogger("uvicorn.access")


def _is_hidden_or_system_file(file_path: Path) -> bool:
    """Ignore metadata/system files accidentally unpacked from archives."""
    parts = file_path.parts
    name = file_path.name
    if "__MACOSX" in parts:
        return True
    if name in {".DS_Store", "Thumbs.db"}:
        return True
    if name.startswith("._"):
        return True
    return False


class InferenceBackend:
    """Orchestrates storage, file handlers, and optional inference engine."""

    def __init__(
        self,
        storage: Storage,
        handlers: FileHandlers,
        engine: Optional[InferenceEngine] = None
        ):
        self.storage = storage
        self.engine = engine
        self.handlers = handlers



    def accept_file(self, filepath: Path) -> List[FileInfo]:
        """Triggers file pre-processing on the uploaded (ZIP) file.
        Internal content should be extracted, and handled according 
        to the detected file format.

        Returns:
            List[FileInfo] - contains the list of files extracted
            from the given input archive.

        Method throws exceptions on error.            
        """
        
        logger.info(f"InferenceBackend.accept_file - {filepath.as_posix()}")

        return self.handlers.accept_file(filepath, self.storage)



    def get_available_inference_files(self) -> List[FileInfo]:
        """Returns the list of files available for inference.
        
        Returns:
            List[FileInfo] - 
        """

        logger.info(f"InferenceBackend.get_available_inference_files")

        result = []

        # This folder should now only contain supported accepted files!
        tif_path = self.storage.get_folderpath(Storage.TIF_FOLDER)
        files = self.storage.rglob(Storage.TIF_FOLDER, "*")

        for file_path in files:
            if file_path.is_file() and not _is_hidden_or_system_file(file_path):
                result.append(get_file_info(file_path, tif_path))

        return result


    def execute_inference(
        self,
        image_file_path: Path,
        model_name: str,
        model_options: Optional[dict] = None,
    ) -> None:
        """Synchronously execute model inference on the given file."""
        logger.info(f"InferenceBackend.execute_inference - {image_file_path.as_posix()}")
        if self.engine is None:
            raise RuntimeError("Inference backend does not have a valid inference engine.")
        
        # Delegate the call
        self.engine.process(
            image_file_path=image_file_path,
            model_name=model_name,
            storage=self.storage,
            model_options=model_options,
        )



    def get_result_files(self) -> List[FileInfo]:
        """Returns the list of inference results.
        
        Returns:
            List[FileInfo] - files
        """
        
        logger.info(f"InferenceBackend.get_result_files")

        result = []

        # This folder should now only contain results
        tif_path = self.storage.get_folderpath(Storage.ANNOTATION_FOLDER)
        files = self.storage.rglob(Storage.ANNOTATION_FOLDER, "*")

        for file_path in files:
            if file_path.is_file():
                result.append(get_file_info(file_path, tif_path))

        return result

