import os
import uvicorn
from fastapi import FastAPI


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
