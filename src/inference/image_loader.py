import cv2
from pathlib import Path
import logging

from inference.inference_engine import IImageLoader

logger = logging.getLogger("uvicorn.access")



class OpenCvImageLoader(IImageLoader):
    def imread(self, file_path):

        logger.debug(f"OpenCvImageLoader.imread - {file_path.as_posix()}")

        # Standard loader
        image = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

        
