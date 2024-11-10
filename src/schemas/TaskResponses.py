import uuid
from enum import Enum

from pydantic import BaseModel


class AsyncResultStatus(str, Enum):
    """Represents the actual state of the background task."""
    pending = 'PENDING'
    started = 'STARTED'
    retry = 'RETRY'
    failure = 'FAILURE'
    success = 'SUCCESS'


class AsyncTaskResponse(BaseModel):
    """Represents the response containing status of the background task."""
    task_id: uuid.UUID
    status: AsyncResultStatus


class PredictStructureResponse(BaseModel):
    s: str