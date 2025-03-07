import argparse
import sys

from pathlib import Path

from backend.storage import Storage
from backend.inference_backend import InferenceBackend
from backend.file_handlers import FileHandlers
from backend.handlers.zip_handler import ZipArchiveHandler
from backend.handlers.image_handler import ImageFileHandler

from inference.inference_engine import InferenceEngine
from inference.image_loader import OpenCvImageLoader
from inference.iedl.model import IedlModel

import logging


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG, 
        format="%(asctime)s %(levelname)s: %(message)s", 
        stream=sys.stdout
    )
    logging.info("Logging configured")


def main(args):

    setup_logging()

    # Build the backend
    backend = InferenceBackend(
        handlers=FileHandlers([
            ZipArchiveHandler(),
            ImageFileHandler()
        ]),
        storage=Storage(basepath=Path(args.storage)),
        engine=InferenceEngine(
            image_loader=OpenCvImageLoader(),
            models={
                "iedl": IedlModel(
                    models_path=Path(args.storage) / "trained_models"
                )
            }
        )
    )

    # Run the inference
    backend.execute_inference(
        image_file_path=Path(args.filename),
        model_name=args.model
    )





if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--filename", "-f", 
        type=str, help="Filename to be processed"
        )
    p.add_argument(
        "--model", "-m", 
        default="iedl", 
        type=str, help="Model to be used for processing. Default (iedl)"
        )
    p.add_argument(
        "--storage", "-s", 
        default="./iedl_root_dir", 
        help="Base folder for storing backend files"
        )
    
    
    main(p.parse_args())



