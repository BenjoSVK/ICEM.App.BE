"""
Data API endpoints: TIFF/GeoJSON listing, structure prediction, zip upload, task status, download, clear.
"""
import logging
import os
from pathlib import Path
from glob import glob
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse, JSONResponse
from celery.result import AsyncResult
from celery_tasks.process_folder import celery_app, process_tiff_files, unzip_file

from config import get_settings
from schemas.base import User
from schemas.TaskResponses import (
    AsyncTaskResponse,
    PredictStructureRequest,
    PredictStructureResponse,
)
from services.auth import get_current_user
from backend.factory import get_backend
from backend.inference_backend import InferenceBackend
from backend.storage import Storage

logger = logging.getLogger("uvicorn.access")

router = APIRouter()
settings = get_settings()


@router.get("/test")
def test() -> dict:
    """Simple health check: returns status ok when the API is up."""
    return {"status": "ok"}


# this endpoint process list of tiff files with given ids, the ids are in requests as list
@router.post(
    "/predict_structure", response_model=PredictStructureResponse, status_code=200
)
async def predict_structure(
    body: PredictStructureRequest,
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
) -> PredictStructureResponse:
    tiff_ids = body.tiff_ids

    logger.info(
        f"Predicting structure for tiff ids: {tiff_ids}, from user: {current_user.username}"
    )
    try:

        # OK - this is ugly!
        base_path = backend.storage.get_folderpath(Storage.TIF_FOLDER)
        available = backend.get_available_inference_files()

        # IDs are file stems.
        available_ids = [
            (Path(fi.id).stem.split('.')[0], base_path / fi.id)
            for fi in available
        ]

        file_paths = []
        for id, filepath in available_ids:
            if id in tiff_ids:
                file_paths.append(filepath.as_posix())

        # The list to process
        if len(file_paths) == 0:
            raise HTTPException(status_code=404, detail="No tiff files found")

        result = process_tiff_files.delay({"file_paths": file_paths})
        logger.info(f"Task id: {result.id}")

        return PredictStructureResponse(
            message="Processing tiff files started",
            incorrect_tiff_ids=[],
            task_id=result.id,
            tiff_files=file_paths,
        )

    except Exception as e:
        logger.exception("Error processing predict_structure request")
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}"
        )


@router.post("/upload_zip", response_model=AsyncTaskResponse, status_code=200)
async def transfer_zip_data(
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
    zipFolder: UploadFile = File(...),
) -> AsyncTaskResponse:
    # Log as soon as upload request is received (before reading body)
    logger.info(
        f"Uploading file: {zipFolder.filename}, from user: {current_user.username}"
    )
    try:

        # Copy the given file to the ZIP folder    
        target_filename = backend.storage.get_filepath(
            Storage.ZIP_FOLDER, 
            zipFolder.filename,
            create_parents=True
        )

        # Save the uploaded file into the ZIP_FOLDER
        logger.info(
            f"Receiving file content: {zipFolder.filename}, from user: {current_user.username}"
        )
        with open(f"{target_filename.as_posix()}", "wb") as f:
            while contents := await zipFolder.read(1024 * 1024):
                f.write(contents)

        # Accept the file by the backend (via Celery)
        result = unzip_file.delay({ "file_path": target_filename.as_posix() })

        logger.info(f"Task id: {result.id}")

        logger.info(
            f"File uploaded successfully: {zipFolder.filename}, from user: {current_user.username}"
        )
        return AsyncTaskResponse(
            message="Data transferred successfully",
            task_id=result.id,
        )

    except Exception as e:
        logger.exception("Error processing upload_zip request")
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}"
        )


@router.get("/task-status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
) -> JSONResponse:
    """Return Celery task status (Pending, Success, or Failed) and result if completed."""
    logger.info(
        f"Getting task status for task id: {task_id}, from user: {current_user.username}"
    )
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == "PENDING":
        return JSONResponse(
            content={"status": "Pending", "task_id": task_id, "result": None},
            status_code=200,
        )
    if task_result.state in ("STARTED", "PROGRESS"):
        # Task is still running; FE shows "Processing" and keeps polling
        return JSONResponse(
            content={"status": "Pending", "task_id": task_id, "result": None},
            status_code=200,
        )
    if task_result.state == "SUCCESS":
        result = task_result.get()
        return JSONResponse(
            content={"status": "Success", "task_id": task_id, "result": result},
            status_code=200,
        )
    # FAILURE, RETRY, or any other terminal failure
    return JSONResponse(
        content={"status": "Failed", "task_id": task_id, "result": None},
        status_code=200,
    )


@router.get("/get-tiff-files")
async def get_tiff_files(
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
) -> JSONResponse:
    """List all TIFF files available for inference (id, last_modified, size_bytes)."""
    logger.info(f"Getting tiff files for user: {current_user.username}")

    # List what we've got
    files = backend.get_available_inference_files()

    list_files = []
    for file_info in files:
        list_files.append({
            "id": file_info.id, 
            "last_modified": file_info.last_modified,
            "size_bytes": file_info.size_bytes
        })

    logger.info(f"Found {len(list_files)} tiff files for user: {current_user.username}")

    return JSONResponse(
        content={"tiff_files": list_files},
        status_code=200,
    )


@router.get("/get-geojson-files")
async def get_geojson_files(
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
) -> JSONResponse:
    """List all GeoJSON result files (id, last_modified, size_bytes)."""
    logger.info(f"Getting geojson files for user: {current_user.username}")

    # List what we've got
    files = backend.get_result_files()

    list_files = []
    for file_info in files:
        list_files.append({
            "id": file_info.id, 
            "last_modified": file_info.last_modified,
            "size_bytes": file_info.size_bytes
        })

    logger.info(f"Found {len(list_files)} geojson files for user: {current_user.username}")

    return JSONResponse(
        content={"geojson_files": list_files},
        status_code=200,
    )


@router.get("/download_geojson/{tiff_id}")
async def download_file(
    tiff_id: str,
    type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
) -> FileResponse:
    """Download GeoJSON file for the given tiff_id (optionally filtered by type)."""
    logger.info(
        f"Downloading geojson file for tiff id: {tiff_id} and type: {type}, from user: {current_user.username}"
    )

    tiff_folder = f"{settings.iedl_root_dir}/annotation_folder"
    file_paths = glob(f"{tiff_folder}/{tiff_id}")

    if not file_paths:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = file_paths[0]
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=file_path.split("/")[-1],
    )


@router.delete("/clear-tiff-data/{tiff_id}")
async def clear_tiff_data(
    tiff_id: str,
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
) -> JSONResponse:
    """Remove all files (TIFF, masks, results, annotations) associated with the given tiff_id."""
    logger.info(
        f"Clearing tiff data for tiff id: {tiff_id}, from user: {current_user.username}"
    )
    tiff_folder = f"{settings.iedl_root_dir}/tiff_folder"
    cell_mask_folder = f"{settings.iedl_root_dir}/cell_mask_folder"
    result_folder = f"{settings.iedl_root_dir}/result_folder"
    bg_mask_folder = f"{settings.iedl_root_dir}/bg_mask_folder"
    annotation_folder = f"{settings.iedl_root_dir}/annotation_folder"

    tiff_files = glob(f"{tiff_folder}/{tiff_id}*.tif*")
    cell_mask_files = glob(f"{cell_mask_folder}/*{tiff_id}*.npy")
    result_files = glob(f"{result_folder}/*{tiff_id}*.geojson")
    bg_mask_files = glob(f"{bg_mask_folder}/*{tiff_id}*.npy")
    annotation_files = glob(f"{annotation_folder}/*{tiff_id}*.geojson")

    all_files = (
        tiff_files + cell_mask_files + result_files + bg_mask_files + annotation_files
    )

    logger.info(
        f"Deleting {len(all_files)} files for tiff id: {tiff_id}, from user: {current_user.username}"
    )
    logger.info(f"Files: {all_files}")

    if all_files:
        for file in all_files:
            os.remove(file)

    return JSONResponse(
        content={"message": "Tiff data cleared successfully"},
        status_code=200,
    )
