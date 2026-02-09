import os
import logging
from celery import Celery
from pathlib import Path


from backend.inference_backend import InferenceBackend
from backend.factory import BackendFactory, create_inference_backend, get_backend

logger = logging.getLogger(__name__)


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


@celery_app.task(name="celery_tasks.process_folder.predict_structure", acks_late=True)
def process_tiff_files(details):

    # Get our backend
    backend: InferenceBackend = get_backend()

    files = details["file_paths"]
    for file_path in files:
        path = Path(file_path)
        try:
            backend.execute_inference(
                image_file_path=path,
                model_name="iedl"
            )
        except ValueError as e:
            if "Could not load image from" in str(e):
                logger.warning(
                    "File not found or unreadable, skipping: %s",
                    path.as_posix(),
                    exc_info=False,
                )
            else:
                raise

    return {"result": "success"}


@celery_app.task(name="celery_tasks.process_folder.unzip_file", acks_late=True)
def unzip_file(details):

    # Get our backend
    backend: InferenceBackend = get_backend()

    # Accept by the backend
    backend.accept_file(filepath=Path(details["file_path"]))

    # Probably not necessary to do anything else..
    return {"result": "success"}

