import os
import uvicorn
from fastapi import FastAPI
import torch


from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


from api.api import api_router
from config import get_settings

settings = get_settings()

# on startup check if iedl_root_dir exist


middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(middleware=middleware)

app.include_router(api_router, prefix="/ikem_api")


# app before startup
@app.on_event("startup")
async def startup_event():
    print("Checking if iedl_root_dir exist")
    if not os.path.exists(settings.iedl_root_dir):
        os.makedirs(settings.iedl_root_dir)
        print(f"Created {settings.iedl_root_dir} directory")

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


# app before startup
@app.on_event("startup")
def check_visible_gpu():
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
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
