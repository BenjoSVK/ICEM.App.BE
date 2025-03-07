import logging

import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from backend.storage import Storage
from typing import Dict


logger = logging.getLogger("uvicorn.access")


class IImageLoader(ABC):
    @abstractmethod
    def imread(self, file_path: Path) -> np.ndarray:
        pass


class IInferenceModel(ABC):
    @abstractmethod
    def lazy_initialize(self):
        pass

    @abstractmethod
    def process_file(
        self, 
        image: np.ndarray,
        image_file_path: Path, 
        storage: Storage
    ):
        pass




class InferenceEngine:
    def __init__(
        self, 
        image_loader: IImageLoader,
        models: Dict[str, IInferenceModel],
    ):
        self.image_loader = image_loader
        # Models we can use for inference
        self.models = models



    def process(
        self,
        image_file_path: Path,
        model_name: str,
        storage: Storage
        ):
        """Synchronously executed model inference on the given file.
        Once finished, the model should produce result files
        in the ANNOTATION_FOLDER on storage.

        Returns:
            Nothing if all went well.

        Throws exception is something went wrong.
        """

        logger.info(f"InferenceEngine.process - {image_file_path.as_posix()} (model: {model_name})")

        if not model_name in self.models:
            logger.error(f"Invalid inference model specified: {model_name}")
            raise ValueError(f"Invalid inference model specified: {model_name}")
        
        # Run inference with this model
        model = self.models[model_name]
        model.lazy_initialize()

        # Load the file, and go !
        image = self.image_loader.imread(image_file_path)

        # Process with the model
        model.process_file(image, image_file_path, storage)

