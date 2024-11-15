from multiprocessing.pool import AsyncResult
import os
import uuid
import asyncio
from glob import glob

from fastapi import APIRouter, File, UploadFile
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from celery_tasks.process_folder import unzip_file
from schemas.TaskResponses import AsyncTaskResponse, PredictStructureResponse
from config import get_settings

router = APIRouter()
settings = get_settings()


# this endpoint process list of tiff files with given ids, the ids are in requests as list
@router.post(
    "/predict_structure", response_model=PredictStructureResponse, status_code=200
)
async def predict_structure(tiff_ids: list[str]) -> PredictStructureResponse:
    try:
        tiff_folder = f"{settings.iedl_root_dir}/tiff_folder"

        # these tiff files will be processed by celery task
        tiff_files = []
        for tiff_id in tiff_ids:
            tiff_files.extend(glob(f"{tiff_folder}/{tiff_id}*.tiff"))

        if tiff_files == []:
            return JSONResponse(
                content={"message": "No tiff files found"}, status_code=404
            )

        result = predict_structure.delay(tiff_ids)

        return JSONResponse(
            content={
                "message": "Data transferred successfully",
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
async def transfer_zip_data(zipFolder: UploadFile = File(...)) -> AsyncTaskResponse:

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
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.state == "PENDING":
        return {"status": "Pending"}
    elif task_result.state == "SUCCESS":
        return {"status": "Success", "result": task_result.result}
    else:
        return {"status": task_result.state}
