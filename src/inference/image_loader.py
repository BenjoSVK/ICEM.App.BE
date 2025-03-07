import cv2
from pathlib import Path

from inference.inference_engine import IImageLoader



class OpenCvImageLoader(IImageLoader):
    def imread(self, file_path):
        # Standard loader
        image = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

        
