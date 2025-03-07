
from pathlib import Path
from abc import ABC, abstractmethod

from backend.storage import Storage
from typing import Dict


class IInferenceModel(ABC):
    @abstractmethod
    def lazy_initialize(self):
        pass

    @abstractmethod
    def process_file(self, image_file_path: Path, storage: Storage):
        pass




class InferenceEngine:
    def __init__(
        self, 
        models: Dict[str, IInferenceModel]
    ):
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

        if not model_name in self.models:
            raise ValueError(f"Invalid inference model specified: {model_name}")
        
        # Run inference with this model
        model = self.models[model_name]
        model.lazy_initialize()
        model.process_file(image_file_path, storage)

