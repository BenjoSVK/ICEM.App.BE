import os
from celery import Celery
from pathlib import Path


from backend.inference_backend import InferenceBackend
from backend.factory import BackendFactory, create_inference_backend, get_backend


REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
IEDL_ROOT_DIR = os.environ.get("IEDL_ROOT_DIR")


# Initialize backend
BackendFactory().initialize(
    create_inference_backend(
        storage_path=Path(IEDL_ROOT_DIR),
        enable_inference=True
    )
)

celery_app = Celery(
    "my_app",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)


@celery_app.task(name="celery_tasks.process_folder.predict_structure")
def process_tiff_files(details):

    # Get our backend
    backend: InferenceBackend = get_backend()

    files = details["file_paths"]
    for file_path in files:

        # Run inference on the given file
        backend.execute_inference(
            image_file_path=Path(file_path),
            model_name="iedl"
        )

    return {"result": "success"}


@celery_app.task(name="celery_tasks.process_folder.unzip_file")
def unzip_file(details):

    # Get our backend
    backend: InferenceBackend = get_backend()

    # Accept by the backend
    backend.accept_file(filepath=Path(details["file_path"]))

    # Probably not necessary to do anything else..
    return {"result": "success"}

