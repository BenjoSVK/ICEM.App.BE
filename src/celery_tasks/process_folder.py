import os
import logging
from celery import Celery
from pathlib import Path

from services.iiif_cache_warm import warm_iiif_cache_for_tiff_id
from services.result_bundle import build_result_bundle
from services.file_registry import upsert_user_tiff_files
from backend.inference_backend import InferenceBackend
from backend.factory import BackendFactory, create_inference_backend, get_backend
from db_handler import get_session

logger = logging.getLogger(__name__)

for _vips_log in ("pyvips", "VIPS"):
    logging.getLogger(_vips_log).setLevel(logging.WARNING)

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
    model_name = details.get("model_name", "iedl")
    cell_model_id = details.get("cell_model_id")
    for file_path in files:
        path = Path(file_path)
        try:
            backend.execute_inference(
                image_file_path=path,
                model_name=model_name,
                model_options={"cell_model_id": cell_model_id} if cell_model_id else None,
            )
            build_result_bundle(backend.storage, path)
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


@celery_app.task(name="celery_tasks.process_folder.warm_iiif_cache", acks_late=True)
def warm_iiif_cache_task(details: dict):
    """Pre-gen iiif_cache for whole TIFF (id is relative for tiff_folder)."""

    return warm_iiif_cache_for_tiff_id(details["tiff_id"])


@celery_app.task(name="celery_tasks.process_folder.unzip_file", acks_late=True)
def unzip_file(details):

    # Get our backend
    backend: InferenceBackend = get_backend()

    # Accept by the backend
    files = backend.accept_file(filepath=Path(details["file_path"]))
    username = details.get("username")
    if username:
        db = get_session()
        try:
            upsert_user_tiff_files(db, username, files)
        finally:
            db.close()

    if os.environ.get("IIIF_WARM_ON_IMPORT").lower() == "1":
        for fi in files:
            if fi.id.lower().endswith((".tif", ".tiff")):
                warm_iiif_cache_task.delay({"tiff_id": fi.id})

    return {"result": "success", "files": len(files)}

