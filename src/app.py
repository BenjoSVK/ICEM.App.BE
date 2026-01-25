"""
This module contains the FastAPI application for the backend service.
"""
import os
import torch
import uvicorn
import logging

from pathlib import Path
from fastapi import FastAPI


from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


from api.api import api_router
from config import get_settings

from backend.factory import create_inference_backend, BackendFactory

settings = get_settings()

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"], # Change this to the specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

# Initialize backend
BackendFactory().initialize(
    create_inference_backend(
        storage_path=Path(settings.iedl_root_dir),
        enable_inference=False
    )
)

app = FastAPI(middleware=middleware)

app.include_router(api_router, prefix="/ikem_api")


# app before startup
@app.on_event("startup")
async def startup_event():
    """
    Check if they exist and if not
    create necessary folders and setup logging.
    """
        # Create iedl_root_dir if not exist
    print("Checking if iedl_root_dir exist")
    if not os.path.exists(settings.iedl_root_dir):
        os.makedirs(settings.iedl_root_dir)
        print(f"Created {settings.iedl_root_dir} directory")

        # create log file if not exist
    log_file = os.path.join(settings.iedl_root_dir, "ikem.log")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Created log file")

    logger = logging.getLogger("uvicorn.access")
    logger.handlers.clear()  # Clear existing handlers

    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    # File logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Ensure the logger uses only your handlers
    logger.propagate = False

    # zip folder
    zip_folder = os.path.join(settings.iedl_root_dir, "zip_folder")
    if not os.path.exists(zip_folder):
        os.makedirs(zip_folder)
        print(f"Created {zip_folder} directory")

    # tiff folder
    tiff_folder = os.path.join(settings.iedl_root_dir, "tiff_folder")
    if not os.path.exists(tiff_folder):
        os.makedirs(tiff_folder)
        print(f"Created {tiff_folder} directory")

    # result folder
    result_folder = os.path.join(settings.iedl_root_dir, "result_folder")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Created {result_folder} directory")

    # bg mask folder
    bg_mask_folder = os.path.join(settings.iedl_root_dir, "bg_mask_folder")
    if not os.path.exists(bg_mask_folder):
        os.makedirs(bg_mask_folder)
        print(f"Created {bg_mask_folder} directory")

    # cell mask folder
    cell_mask_folder = os.path.join(settings.iedl_root_dir, "cell_mask_folder")
    if not os.path.exists(cell_mask_folder):
        os.makedirs(cell_mask_folder)
        print(f"Created {cell_mask_folder} directory")

    # annotation folder
    annotation_folder = os.path.join(settings.iedl_root_dir, "annotation_folder")
    if not os.path.exists(annotation_folder):
        os.makedirs(annotation_folder)
        print(f"Created {annotation_folder} directory")


# app before startup
@app.on_event("startup")
def check_visible_gpu():
    """
    Check if GPU is available and print device info.
    """
    # print visible devices
    if torch.cuda.is_available():
        print("Visible devices: ", torch.cuda.device_count())
        print("Current device: ", torch.cuda.current_device())
        print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("No GPU available")


def main():
    """
    Run the backend app using uvicorn.
    """
    uvicorn.run(
        "app:app",
        reload=settings.reload,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
    )


if __name__ == "__main__":
    main()
