import os
import uuid

from fastapi import APIRouter, File, UploadFile
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.celery_tasks.process_folder import process_tiff_prediction
from src.schemas.TaskResponses import AsyncTaskResponse, PredictStructureResponse

router = APIRouter()


@router.get("/test")
async def test():
    return {"message": "Hello World"}


@router.post("/transfer_zip_data")
async def transfer_zip_data(zipFolder: UploadFile = File(...)):

    # Create transfered_data folder if it doesn't exist
    if not os.path.exists("src/transfered_data"):
        os.makedirs("src/transfered_data")

    try:
        with open("src/transfered_data/"+zipFolder.filename, 'wb') as f:
            while contents := await zipFolder.read(1024 * 1024):  # Read in chunks of 1MB
                f.write(contents)
        return JSONResponse(content={"message": "Data transferred successfully"})
    except Exception as e:
        return JSONResponse(content={"message": f"Error processing file: {str(e)}"}, status_code=500)


@router.post(
    '/process_folder',
    response_model=AsyncTaskResponse,
    status_code=202
)
async def transfer_zip_data(zipFolder: UploadFile = File(...)) -> AsyncTaskResponse:

    try:
        # Create transfered_data folder if it doesn't exist
        # Data handling can be changed
        if not os.path.exists("src/transfered_data"):
            os.makedirs("src/transfered_data")

        with open("src/transfered_data/"+zipFolder.filename, 'wb') as f:
            while contents := await zipFolder.read(1024 * 1024):  # Read in chunks of 1MB
                f.write(contents)

        # Perform celery task
        task = process_tiff_prediction.delay({"zip": zipFolder.filename})

        return {'task_id': task.task_id, 'status': task.status}

    except Exception as e:
        return JSONResponse(content={"message": f"Error processing file: {str(e)}"}, status_code=500)
        # return {'task_id': task.task_id, 'status': task.status}


@router.get(
    '/process_folder',
    response_model=PredictStructureResponse,
    responses={
        202: {'model': AsyncTaskResponse}
    }
)
async def get_prediction_result(task_id: uuid.UUID) -> PredictStructureResponse:
    pass