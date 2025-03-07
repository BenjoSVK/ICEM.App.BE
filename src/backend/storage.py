
from pathlib import Path
import shutil
import tempfile


class Storage:

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
        subfolder: str=None,
        create_parents: bool=True
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
        create_parents: bool=False
    ) -> Path:
        """Compose path components for target filename which 
        should be located in the target folder and create all
        parent folders.
        """
        filepath = self.basepath / folder / filename
        if create_parents:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath
    

    def rglob(self, folder, pattern):
        """Returns all files from the folder"""
        return sorted(list((self.basepath / folder).rglob(pattern)))
    
    def get_temp_folder(self):

        temp = self.basepath / "temp"
        temp.mkdir(parents=True, exist_ok=True)

        # New temporary folder under basepath/temp
        return TemporaryFolder(temp)




class TemporaryFolder:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.temp_dir = None

    def __enter__(self):
        self.temp_dir = Path(tempfile.mkdtemp(dir=self.base_path))
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir.as_posix())


