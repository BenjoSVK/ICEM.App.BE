from enum import Enum

from pydantic import BaseModel, Field


class AsyncResultStatus(str, Enum):
    """Represents the actual state of the background task."""
    pending = 'PENDING'
    started = 'STARTED'
    retry = 'RETRY'
    failure = 'FAILURE'
    success = 'SUCCESS'


class AsyncTaskResponse(BaseModel):
    """Response for async operations (e.g. upload_zip) returning task id."""
    message: str
    task_id: str


class PredictStructureRequest(BaseModel):
    """Request body for predict_structure - list of tiff IDs must not be empty."""
    tiff_ids: list[str] = Field(..., min_length=1, description="Non-empty list of tiff file IDs")


class PredictStructureResponse(BaseModel):
    """Response when predict_structure successfully starts processing."""
    message: str
    incorrect_tiff_ids: list[str] = []
    task_id: str
    tiff_files: list[str]