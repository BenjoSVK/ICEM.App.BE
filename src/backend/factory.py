"""
Backend factory: creates InferenceBackend and holds the singleton used by API dependencies.
"""
from pathlib import Path
from typing import Optional

from backend.file_handlers import FileHandlers
from backend.handlers.image_handler import ImageFileHandler
from backend.handlers.zip_handler import ZipArchiveHandler
from backend.inference_backend import InferenceBackend
from backend.storage import Storage
from inference.image_loader import OpenCvImageLoader
from inference.iedl.model import IedlModel
from inference.inference_engine import InferenceEngine


def create_inference_backend(
    storage_path: Path,
    enable_inference: bool,
) -> InferenceBackend:
    """Build InferenceBackend with storage, file handlers, and optionally an inference engine."""
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
    """Singleton that holds the single InferenceBackend instance for the app."""

    _instance: Optional["BackendFactory"] = None

    def __new__(cls) -> "BackendFactory":
        if cls._instance is None:
            cls._instance = super(BackendFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "backend"):
            self.backend: Optional[InferenceBackend] = None

    def initialize(self, backend: InferenceBackend) -> None:
        """Set the backend instance (call once at startup)."""
        self.backend = backend


def get_backend() -> InferenceBackend:
    """FastAPI dependency: return the initialized InferenceBackend or raise RuntimeError."""
    backend = BackendFactory().backend
    if backend is None:
        raise RuntimeError(
            "Inference backend is not initialized. Call BackendFactory().initialize(backend) first."
        )
    return backend



