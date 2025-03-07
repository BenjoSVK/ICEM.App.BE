from pathlib import Path
from typing import List, Optional
from datetime import datetime
import zipfile

from backend.storage import Storage
from inference.inference_engine import InferenceEngine
from schemas.file_info import FileInfo

class InferenceBackend:
    def __init__(
        self,
        storage: Storage,
        engine: Optional[InferenceEngine] = None
        ):
        self.storage = storage
        self.engine = engine



    def accept_file(self, filepath: Path) -> List[FileInfo]:
        """Triggers file pre-processing on the uploaded (ZIP) file.
        Internal content should be extracted, and handled according 
        to the detected file format.

        Returns:
            List[FileInfo] - contains the list of files extracted
            from the given input archive.

        Method throws exceptions on error.            
        """
        result = []
        
        if not filepath.exists():
            raise FileNotFoundError(f"Input archive file '{filepath}' not found.")
        
        if filepath.suffix.lower() != ".zip":
            raise ValueError(
                f"Invalid file type: expected .zip, but got '{filepath.suffix.lower()}'"
            )

        # We create subfolder for this job
        job_name = filepath.stem        
        target_folder = self.storage.get_folderpath(
            folder=Storage.TIF_FOLDER, 
            subfolder=job_name
        )

        # Extract the archive
        with zipfile.ZipFile(filepath, "r") as archive:
            archive.extractall(target_folder)

        # Filter the extracted files
        to_delete = []
        to_accept = []
        extracted_files = sorted(list(target_folder.rglob("*")))
        for f in extracted_files:
            if self._is_valid_image_file(f):
                to_accept.append(f)
            else:
                to_delete.append(f)

        # Delete files marked as invalid
        for f in to_delete:
            self._recursive_delete(f)

        # Return a list of files we wish to keep
        tif_path = self.storage.get_folderpath(Storage.TIF_FOLDER)
        result = []
        for file_path in to_accept:
            result.append(
                self._get_file_info(file_path, tif_path)
            )

        return result



    def get_available_inference_files(self) -> List[FileInfo]:
        """Returns the list of files available for inference.
        
        Returns:
            List[FileInfo] - 
        """

        # In the base implementation we are interested only 
        # in TIF files located in the TIF_FOLDER

        result = []

        # TODO: Add some other files as well ?
        tif_path = self.storage.get_folderpath(Storage.TIF_FOLDER)
        files = self.storage.rglob(Storage.TIF_FOLDER, "*.tif*")

        for file_path in files:
            result.append(
                self._get_file_info(file_path, tif_path)
            )

        return result


    def execute_inference(
        self, 
        image_file_path: Path,
        model_name: str
    ):
        """Synchronously execute model inference on the given file."""
        if self.engine is None:
            raise RuntimeError("Inference backend does not have a valid inference engine.")
        
        # Delegate the call
        self.engine.process(
            image_file_path=image_file_path,
            model_name=model_name,
            storage=self.storage
        )



    def get_result_files(self) -> List[FileInfo]:
        """Returns the list of inference results.
        
        Returns:
            List[FileInfo] - files
        """

        result = []



    def _get_file_info(
        self, 
        file_path: Path,
        base_path: Path
    ) -> FileInfo:
        """Returns a FileInfo for the given file path"""
        last_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)    
        return FileInfo(
            id=file_path.relative_to(base_path).as_posix(),
            last_modified=last_modified_time.strftime("%Y-%m-%d"),
            size_bytes=file_path.stat().st_size
        )


    def _is_valid_image_file(self, file_path: Path) -> bool:
        """Decides wheather a file is to be kept or not"""
        return True
    

    def _recursive_delete(self, path: Path):
        if path.exists():       
            if path.is_dir():
                for subpath in path.glob("*"):
                    self._recursive_delete(subpath)
                path.rmdir()

            if path.is_file():
                path.unlink()