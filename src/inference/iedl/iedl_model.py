
from inference.inference_engine import IInferenceModel


class IedlModel(IInferenceModel):
    def __init__(self):
        pass


    def lazy_initialize(self):
        """Load pytorch models and stuff"""
        pass


    def process_file(self, image_file_path, storage):
        """Lets do the shit!"""

        pass


    
        