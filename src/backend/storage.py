"""
Storage paths and temp folder helpers under a configurable base path.
"""
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional


class Storage:
    """Manages folder and file paths under a base directory (ZIP, TIFF, results, etc.)."""

    ZIP_FOLDER = "zip_folder"
    TIF_FOLDER = "tiff_folder"
    RESULT_FOLDER = "result_folder"
    BACKGROUND_MASK_FOLDER = "bg_mask_folder"
    CELL_MASK_FOLDER = "cell_mask_folder"
    ANNOTATION_FOLDER = "annotation_folder"

    def __init__(
        self,
        basepath: Path
    ):        
        self.basepath = basepath
        

    def get_folderpath(
        self,
        folder: str,
        subfolder: Optional[str] = None,
        create_parents: bool = True,
    ) -> Path:
        """Compose path components and create target folder."""
        result = self.basepath / folder
        if subfolder is not None:
            result = result / subfolder
        if create_parents:
            result.mkdir(parents=True, exist_ok=True)
        return result


    def get_filepath(
        self,
        folder: str,
        filename: str,
        create_parents: bool = False,
    ) -> Path:
        """Compose path for a file in the given folder; optionally create parent dirs."""
        filepath = self.basepath / folder / filename
        if create_parents:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath
    

    def rglob(self, folder: str, pattern: str) -> List[Path]:
        """Return sorted paths of all files under folder matching the given pattern."""
        return sorted((self.basepath / folder).rglob(pattern))

    def get_temp_folder(self) -> "TemporaryFolder":
        """Create and return a temporary folder context under basepath/temp."""
        temp = self.basepath / "temp"
        temp.mkdir(parents=True, exist_ok=True)
        return TemporaryFolder(temp)




class TemporaryFolder:
    """Context manager for a temporary directory under a given base path."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.temp_dir: Optional[Path] = None

    def __enter__(self) -> Path:
        self.temp_dir = Path(tempfile.mkdtemp(dir=self.base_path))
        return self.temp_dir

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir.as_posix())


