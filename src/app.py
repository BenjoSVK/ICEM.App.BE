"""
This module contains the FastAPI application for the backend service.
"""
import os
import torch
import uvicorn
import logging
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI


from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from api.api import api_router
from api.limiter import limiter
from config import get_settings

from backend.factory import create_inference_backend, BackendFactory

settings = get_settings()

_cors_origins = settings.cors_origins
_cors_credentials = "*" not in _cors_origins

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_credentials,
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create dirs, setup logging, print GPU info. Shutdown: (none)."""

    if not os.path.exists(settings.iedl_root_dir):
        os.makedirs(settings.iedl_root_dir)
        print(f"INFO:     Created {settings.iedl_root_dir} directory")

    log_file = os.path.join(settings.iedl_root_dir, "ikem.log")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("INFO:     Created log file")

    logger = logging.getLogger("uvicorn.access")
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.propagate = False

    for folder_name in (
        "zip_folder", "tiff_folder", "result_folder", "bg_mask_folder",
        "cell_mask_folder", "annotation_folder",
    ):
        folder = os.path.join(settings.iedl_root_dir, folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"INFO:     Created {folder} directory")

    if torch.cuda.is_available():
        print("INFO:     Visible devices: ", torch.cuda.device_count())
        print("INFO:     Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("INFO:     No GPU available")
    yield


app = FastAPI(middleware=middleware, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(api_router, prefix="/ikem_api")


def main() -> None:
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