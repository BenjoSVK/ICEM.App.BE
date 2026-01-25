
from pathlib import Path

from backend.storage import Storage
from backend.inference_backend import InferenceBackend
from backend.file_handlers import FileHandlers
from backend.handlers.zip_handler import ZipArchiveHandler
from backend.handlers.image_handler import ImageFileHandler

from inference.inference_engine import InferenceEngine
from inference.image_loader import OpenCvImageLoader
from inference.iedl.model import IedlModel


def create_inference_backend(
    storage_path: Path, 
    enable_inference: bool
) -> InferenceBackend:
    
    # Decide if inference is necessary
    engine = None
    if enable_inference:
        engine = InferenceEngine(
            image_loader=OpenCvImageLoader(),
            models={
                "iedl": IedlModel(
                    models_path=storage_path / "trained_models"
                )
            }
        )

    # Build the backend
    backend = InferenceBackend(
        handlers=FileHandlers([
            ZipArchiveHandler(),
            ImageFileHandler()
        ]),
        storage=Storage(basepath=storage_path),
        engine=engine
    )

    return backend



class BackendFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackendFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "backend"):
            self.backend = None

    def initialize(self, backend):
        self.backend = backend


# Single backend instance !
def get_backend():
    return BackendFactory().backend



