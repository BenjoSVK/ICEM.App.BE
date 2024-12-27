from datetime import datetime
from celery.result import AsyncResult
import os
from glob import glob

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from celery_tasks.process_folder import unzip_file, process_tiff_files
from schemas.TaskResponses import AsyncTaskResponse, PredictStructureResponse
from config import get_settings
from api.endpoints.auth import (
    User,
    get_current_user,
)

router = APIRouter()
settings = get_settings()


# this endpoint process list of tiff files with given ids, the ids are in requests as list
@router.post(
    "/predict_structure", response_model=PredictStructureResponse, status_code=200
)
async def predict_structure(
    tiff_ids: list[int],
    current_user: User = Depends(get_current_user),
) -> PredictStructureResponse:
    try:
        tiff_folder = f"{settings.iedl_root_dir}/tiff_folder"

        # these tiff files will be processed by celery task
        tiff_files = []
        for tiff_id in tiff_ids:
            tiff_files.extend(glob(f"{tiff_folder}/{tiff_id}*.tif*"))

        if tiff_files == []:
            return JSONResponse(
                content={"message": "No tiff files found"}, status_code=404
            )

        details = {
            "tiff_files": tiff_files,
            "tiff_folder": tiff_folder,
            "bg_mask_folder": f"{settings.iedl_root_dir}/bg_mask_folder",
            "cell_mask_folder": f"{settings.iedl_root_dir}/cell_mask_folder",
            "id_list": tiff_ids,
        }
        result = process_tiff_files.delay(details)

        return JSONResponse(
            content={
                "message": "Processing tiff files started",
                "task_id": result.id,
                "tiff_files": tiff_files,
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
    zipFolder: UploadFile = File(...),
) -> AsyncTaskResponse:

    print("uploading zip file")
    try:
        zip_folder = f"{settings.iedl_root_dir}/zip_folder"

        # save zip file into zip folder
        with open(f"{zip_folder}/{zipFolder.filename}", "wb") as f:
            while contents := await zipFolder.read(1024 * 1024):
                f.write(contents)

        # process zip file
        result = unzip_file.delay(
            {
                "zip_path": f"{zip_folder}/{zipFolder.filename}",
                "tiff_folder": f"{settings.iedl_root_dir}/tiff_folder",
            }
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
    task_result = AsyncResult(task_id)
    if task_result.state == "PENDING":
        return JSONResponse(
            content={"status": "Pending", "task_id": task_id}, status_code=200
        )
    elif task_result.state == "SUCCESS":
        return JSONResponse(
            content={"status": "Success", "task_id": task_id}, status_code=200
        )
    else:
        return JSONResponse(
            content={"status": "Failed", "task_id": task_id},
            status_code=200,
        )


# get every tiff file in the tiff folder
@router.get("/get-tiff-files")
async def get_tiff_files(
    current_user: User = Depends(get_current_user),
):
    tiff_folder = f"{settings.iedl_root_dir}/tiff_folder"
    tiff_files = glob(f"{tiff_folder}/*.tif*")

    files_info = []
    for file in tiff_files:
        file_name = os.path.basename(file)
        file_id = os.path.splitext(file_name)[0]
        mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d")
        file_size = os.path.getsize(file) / (1024 * 1024)  # Get file size in MB

        files_info.append(
            {"id": file_id, "last_modified": mod_time, "size_bytes": file_size}
        )

    return JSONResponse(
        content={"tiff_files": files_info},
        status_code=200,
    )


# get geojson files
@router.get("/get-geojson-files")
async def get_geojson_files(
    current_user: User = Depends(get_current_user),
):
    geojson_folder = f"{settings.iedl_root_dir}/result_folder"
    geojson_files = glob(f"{geojson_folder}/*.geojson")

    files_info = []
    for file in geojson_files:
        file_name = os.path.basename(file)
        file_id = os.path.splitext(file_name)[0]
        mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d")
        file_size = os.path.getsize(file) / (1024 * 1024)  # Get file size in MB

        files_info.append(
            {"id": file_id, "last_modified": mod_time, "size_bytes": file_size}
        )

    return JSONResponse(
        content={"geojson_files": files_info},
        status_code=200,
    )


@router.get("/download_geojson/{tiff_id}")
async def download_file(
    tiff_id: str,
    current_user: User = Depends(get_current_user),
):  # Changed id to str
    tiff_folder = f"{settings.iedl_root_dir}/result_folder"
    file_paths = glob(f"{tiff_folder}/{tiff_id}*.geojson")

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
):
    tiff_folder = f"{settings.iedl_root_dir}/tiff_folder"
    cell_mask_folder = f"{settings.iedl_root_dir}/cell_mask_folder"
    result_folder = f"{settings.iedl_root_dir}/result_folder"
    bg_mask_folder = f"{settings.iedl_root_dir}/bg_mask_folder"

    tiff_files = glob(f"{tiff_folder}/{tiff_id}*.tif*")
    cell_mask_files = glob(f"{cell_mask_folder}/*{tiff_id}*.npy")
    result_files = glob(f"{result_folder}/*{tiff_id}*.geojson")
    bg_mask_files = glob(f"{bg_mask_folder}/*{tiff_id}*.npy")

    all_files = tiff_files + cell_mask_files + result_files + bg_mask_files

    if all_files:
        for file in all_files:
            os.remove(file)

    return JSONResponse(
        content={"message": "Tiff data cleared successfully"},
        status_code=200,
    )
