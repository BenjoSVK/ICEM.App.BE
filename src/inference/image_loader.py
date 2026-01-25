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
        
        # Check if image was loaded successfully
        if image is None:
            logger.error(f"Failed to load image from {file_path.as_posix()}")
            raise ValueError(f"Could not load image from {file_path.as_posix()}")
       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

        
