from datetime import datetime
from celery.result import AsyncResult
import os
from glob import glob
import logging
import re
from pathlib import Path

logger = logging.getLogger("uvicorn.access")

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from celery_tasks.process_folder import unzip_file, process_tiff_files
from schemas.TaskResponses import AsyncTaskResponse, PredictStructureResponse
from config import get_settings
from schemas.base import User

from services.auth import get_current_user

from backend.inference_backend import InferenceBackend
from backend.factory import get_backend
from backend.storage import Storage

router = APIRouter()
settings = get_settings()


# this endpoint process list of tiff files with given ids, the ids are in requests as list
@router.post(
    "/predict_structure", response_model=PredictStructureResponse, status_code=200
)
async def predict_structure(
    tiff_ids: list[str],
    current_user: User = Depends(get_current_user),
) -> PredictStructureResponse:

    logger.info(
        f"Predicting structure for tiff ids: {tiff_ids}, from user: {current_user.username}"
    )
    try:

        # The list to process
        file_paths = tiff_ids
        if len(file_paths) == 0:
            return JSONResponse(
                content={"message": "No tiff files found"}, 
                status_code=404
            )

        result = process_tiff_files.delay({ "file_paths": file_paths })
        logger.info(f"Task id: {result.id}")

        return JSONResponse(
            content={
                "message": "Processing tiff files started",
                "incorrect_tiff_ids": [],
                "task_id": result.id,
                "tiff_files": file_paths,
            },
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(
            content={"message": f"Error processing file: {str(e)}"}, status_code=500
        )


@router.post("/upload_zip", response_model=AsyncTaskResponse, status_code=200)
async def transfer_zip_data(
    current_user: User = Depends(get_current_user),
#    backend: InferenceBackend = Depends(get_backend),
    upload_file: UploadFile = File(...),
) -> AsyncTaskResponse:

    # Get our backend
    backend: InferenceBackend = get_backend()

    logger.info(
        f"Uploading file: {upload_file.filename}, from user: {current_user.username}"
    )
    try:

        # Copy the given file to the ZIP folder    
        target_filename = backend.storage.get_filepath(
            Storage.ZIP_FOLDER, 
            upload_file.filename,
            create_parents=True
        )

        # Save the uploaded file into the ZIP_FOLDER
        with open(f"{target_filename.as_posix()}", "wb") as f:
            while contents := await upload_file.read(1024 * 1024):
                f.write(contents)

        # Accept the file by the backend (via Celery)
        result = unzip_file.delay({ "file_path": target_filename.as_posix() })

        logger.info(f"Task id: {result.id}")

        logger.info(
            f"File uploaded successfully: {upload_file.filename}, from user: {current_user.username}"
        )
        return JSONResponse(
            content={"message": "Data transferred successfully", "task_id": result.id},
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(
            content={"message": f"Error processing file: {str(e)}"}, status_code=500
        )


@router.get("/task-status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    logger.info(
        f"Getting task status for task id: {task_id}, from user: {current_user.username}"
    )
    task_result = AsyncResult(task_id)

    if task_result.state == "PENDING":
        return JSONResponse(
            content={"status": "Pending", "task_id": task_id, "result": None},
            status_code=200,
        )
    elif task_result.state == "SUCCESS":
        result = task_result.get()
        return JSONResponse(
            content={"status": "Success", "task_id": task_id, "result": result},
            status_code=200,
        )
    else:

        return JSONResponse(
            content={"status": "Failed", "task_id": task_id, "result": None},
            status_code=200,
        )


# get every tiff file in the tiff folder
@router.get("/get-tiff-files")
async def get_tiff_files(
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
):
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


# get geojson files
@router.get("/get-geojson-files")
async def get_geojson_files(
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
):
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
    type: str = None,
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
):  # Changed id to str

    # check wtih regex if tiff id is in form <ID>_<ID2>
    if not tiff_id:
        raise HTTPException(status_code=400, detail="Invalid tiff id")

    if not type:
        raise HTTPException(status_code=400, detail="Invalid type")

    if not re.match(r"\d+_\d+", tiff_id):
        raise HTTPException(
            status_code=400, detail="Invalid tiff id, should be in form <ID>_<ID2>"
        )

    logger.info(
        f"Downloading geojson file for tiff id: {tiff_id} and type: {type}, from user: {current_user.username}"
    )
    if type != "tissue" and type != "cell":
        raise HTTPException(
            status_code=400, detail="Invalid type, must be tissue or cell"
        )

    tiff_folder = f"{settings.iedl_root_dir}/annotation_folder"
    file_paths = glob(f"{tiff_folder}/{type}_mask_{tiff_id}*.geojson")

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
):
    if not re.match(r"\d+_\d+", tiff_id):
        raise HTTPException(
            status_code=400, detail="Invalid tiff id, should be in form <ID>_<ID2>"
        )

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
