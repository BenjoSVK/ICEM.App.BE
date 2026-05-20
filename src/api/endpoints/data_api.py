"""
Data API endpoints: TIFF/GeoJSON listing, structure prediction, zip upload, task status, download, clear.
"""
import json
import logging
import os
from pathlib import Path
from glob import glob
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, Depends, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from celery.result import AsyncResult
from starlette.requests import ClientDisconnect
from sqlalchemy.orm import Session
from celery_tasks.process_folder import celery_app, process_tiff_files, unzip_file

from config import get_settings
from db_handler import get_db
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
from inference.iedl.model import IedlModel
from services.file_registry import (
    delete_user_tiff_mappings,
    list_user_tiff_files,
    list_user_tiff_ids,
    mark_task_terminal,
    mark_tiffs_pending,
    upsert_user_tiff_files,
    user_owns_tiff,
)
from services.result_bundle import build_result_bundle

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
    db: Session = Depends(get_db),
) -> PredictStructureResponse:
    tiff_ids = body.tiff_ids
    model_name = body.model_name
    cell_model_id = body.cell_model_id

    logger.info(
        f"Predicting structure for tiff ids: {tiff_ids}, model: {model_name}, "
        f"cell_model_id: {cell_model_id}, from user: {current_user.username}"
    )
    try:
        if cell_model_id is not None:
            supported_models = {"iedl", "iedl_attention_unet"}
            if model_name not in supported_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_name}' does not support cell_model_id override",
                )
            available_ids = {item["id"] for item in IedlModel.get_registered_cell_models()}
            if cell_model_id not in available_ids:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": f"Unsupported cell_model_id '{cell_model_id}'",
                        "available_cell_model_ids": sorted(available_ids),
                    },
                )

        # OK - this is ugly!
        base_path = backend.storage.get_folderpath(Storage.TIF_FOLDER)
        available = backend.get_available_inference_files()
        allowed_ids = list_user_tiff_ids(db, current_user.username)

        # IDs are file stems.
        available_ids = [
            (fi.id.rsplit('.', 1)[0], fi.id, base_path / fi.id)
            for fi in available
        ]

        file_paths = []
        for id_stem, file_id, filepath in available_ids:
            if id_stem in tiff_ids and file_id in allowed_ids:
                file_paths.append(filepath.as_posix())

        # The list to process
        if len(file_paths) == 0:
            raise HTTPException(status_code=404, detail="No tiff files found")

        result = process_tiff_files.delay(
            {
                "file_paths": file_paths,
                "model_name": model_name,
                "cell_model_id": cell_model_id,
            }
        )
        selected_file_ids = [file_id for id_stem, file_id, _ in available_ids if id_stem in tiff_ids and file_id in allowed_ids]
        mark_tiffs_pending(db, current_user.username, selected_file_ids, result.id)
        logger.info(f"Task id: {result.id}")

        return PredictStructureResponse(
            message="Processing tiff files started",
            incorrect_tiff_ids=[],
            task_id=result.id,
            tiff_files=file_paths,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing predict_structure request")
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}"
        )


@router.get("/predict_structure/cell-models")
async def get_predict_structure_cell_models(
    model_name: str = "iedl",
    current_user: User = Depends(get_current_user),
) -> JSONResponse:
    """Return available selectable cell models for the requested inference model."""
    logger.info(
        "Listing cell models for model=%s, user=%s",
        model_name,
        current_user.username,
    )

    if model_name not in {"iedl", "iedl_attention_unet"}:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' does not expose selectable cell models",
        )

    return JSONResponse(
        content={
            "model_name": model_name,
            "cell_models": IedlModel.get_registered_cell_models(),
        },
        status_code=200,
    )

@router.post("/upload_zip", response_model=AsyncTaskResponse, status_code=200)
async def transfer_zip_data(
    request: Request,
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
) -> AsyncTaskResponse:

    logger.info(f"Uploading file from user: {current_user.username}")

    form = None
    target_filename = None

    try:
        form = await request.form()
        zipFolder: UploadFile = form.get("zipFolder")

        if not zipFolder:
            raise HTTPException(status_code=422, detail="No file provided")

        target_filename = backend.storage.get_filepath(
            Storage.ZIP_FOLDER,
            zipFolder.filename,
            create_parents=True
        )

        logger.info(f"Receiving file content: {zipFolder.filename}, from user: {current_user.username}")

        with open(f"{target_filename.as_posix()}", "wb") as f:
            while contents := await zipFolder.read(1024 * 1024):
                f.write(contents)

        result = unzip_file.delay(
            {"file_path": target_filename.as_posix(), "username": current_user.username}
        )
        logger.info(f"Task id: {result.id}")
        logger.info(f"File uploaded successfully: {zipFolder.filename}, from user: {current_user.username}")

        return AsyncTaskResponse(
            message="Data transferred successfully",
            task_id=result.id,
        )
    except ClientDisconnect:
        logger.info(
            "Upload was interrupted by user: %s",
            current_user.username
        )
        # Non-standard but commonly used status for client-closed request.
        return JSONResponse(
            status_code=499,
            content={"detail": "Upload was interrupted by client."},
        )
    except Exception as e:
        logger.exception("Error processing upload_zip request")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if form is not None:
            await form.close()
        # Best-effort cleanup of partially uploaded file on interruption/failure.
        if target_filename and Path(target_filename).exists():
            try:
                if os.path.getsize(target_filename) == 0:
                    os.remove(target_filename)
            except OSError:
                pass


@router.get("/task-status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
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
        mark_task_terminal(db, task_id, "Success")
        return JSONResponse(
            content={"status": "Success", "task_id": task_id, "result": result},
            status_code=200,
        )
    # FAILURE, RETRY, or any other terminal failure
    mark_task_terminal(db, task_id, "Failed")
    return JSONResponse(
        content={"status": "Failed", "task_id": task_id, "result": None},
        status_code=200,
    )


@router.get("/get-tiff-files")
async def get_tiff_files(
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """List all TIFF files available for inference (id, last_modified, size_bytes)."""
    logger.info(f"Getting tiff files for user: {current_user.username}")

    # Refresh only already owned mappings; never auto-assign foreign TIFFs to this user.
    available = backend.get_available_inference_files()
    owned_ids = list_user_tiff_ids(db, current_user.username)
    owned_files = [file_info for file_info in available if file_info.id in owned_ids]
    if owned_files:
        upsert_user_tiff_files(db, current_user.username, owned_files)
    list_files = list_user_tiff_files(db, current_user.username)

    logger.info(f"Found {len(list_files)} Tiff files for user: {current_user.username}")

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

    logger.info(f"Found {len(list_files)} GeoJSON files for user: {current_user.username}")

    return JSONResponse(
        content={"geojson_files": list_files},
        status_code=200,
    )


@router.get("/geojson-overlay/{tiff_id:path}")
async def get_geojson_overlay(
    tiff_id: str,
    variant: Optional[str] = Query(
        default=None,
        description="Optional GeoJSON variant suffix (e.g. 'resnet' or 'attention')",
    ),
    current_user: User = Depends(get_current_user),
) -> JSONResponse:
    """
    GeoJSON for overlay on top of WSI (via OpenSeadragon). Results from GeoJSON
    """
    logger.info(
        "GeoJSON overlay for tiff_id=%s, user=%s",
        tiff_id,
        current_user.username,
    )

    stem = Path(tiff_id).stem
    result_folder = f"{settings.iedl_root_dir}/result_folder"
    annotation_folder = f"{settings.iedl_root_dir}/annotation_folder"
    variant_suffixes = ("resnet", "attention")
    variant_candidates: dict[str, list[str]] = {suffix: [] for suffix in variant_suffixes}
    for folder in (result_folder, annotation_folder):
        if not os.path.isdir(folder):
            continue
        for suffix in variant_suffixes:
            variant_candidates[suffix].extend(glob(f"{folder}/{stem}_{suffix}.geojson"))

    available_variants = sorted(
        suffix for suffix, files in variant_candidates.items() if files
    )
    if not variant and len(available_variants) > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Multiple GeoJSON variants exist for this slide. "
                    "Specify query parameter 'variant'."
                ),
                "available_variants": available_variants,
                "example": f"/ikem_api/geojson-overlay/{tiff_id}?variant={available_variants[0]}",
            },
        )

    candidates: list[str] = []
    normalized_variant = variant.strip().lower() if variant else None
    for folder in (result_folder, annotation_folder):
        if os.path.isdir(folder):
            if normalized_variant:
                candidates.extend(glob(f"{folder}/{stem}_{normalized_variant}.geojson"))
            else:
                candidates.extend(glob(f"{folder}/*{stem}*.geojson"))
    if not candidates:
        if normalized_variant:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No GeoJSON file found for this slide and variant '{normalized_variant}' "
                    "(result_folder / annotation_folder)."
                ),
            )
        raise HTTPException(
            status_code=404,
            detail="No GeoJSON file found for this slide (result_folder / annotation_folder).",
        )
    path = sorted(candidates)[0]
    try:
        raw = Path(path).read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.exception("Failed to read GeoJSON overlay: %s", path)
        raise HTTPException(
            status_code=500, detail=f"Invalid GeoJSON file: {e}"
        ) from e
    return JSONResponse(content=data)


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


@router.get("/download-result/{tiff_id:path}")
async def download_wsi_result(
    tiff_id: str,
    current_user: User = Depends(get_current_user),
    backend: InferenceBackend = Depends(get_backend),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download a prebuilt ZIP bundle with WSI file + related GeoJSON results(s)."""
    logger.info("Downloading WSI results bundle for tiff_id=%s, user=%s", tiff_id, current_user.username)

    if not user_owns_tiff(db, current_user.username, tiff_id):
        raise HTTPException(status_code=403, detail="You do not have access to this TIFF")

    tiff_root = backend.storage.get_folderpath(Storage.TIF_FOLDER)
    tiff_path = (tiff_root / tiff_id).resolve()
    if not tiff_path.exists() or not tiff_path.is_file():
        # Fallback: tolerate callers without extension or with stem.
        stem = Path(tiff_id).stem
        matches = glob(f"{tiff_root.as_posix()}/**/*{stem}*.tif*", recursive=True)
        if not matches:
            raise HTTPException(status_code=404, detail="TIFF file not found")
        tiff_path = Path(sorted(matches)[0]).resolve()

    stem = tiff_path.stem
    result_path = backend.storage.get_filepath(Storage.RESULT_FOLDER, f"{stem}.zip", create_parents=True)
    if not result_path.exists():
        result_path = build_result_bundle(backend.storage, tiff_path)

    return FileResponse(
        result_path.as_posix(),
        media_type="application/zip",
        filename=result_path.name,
    )

# TODO: Implement
@router.delete("/clear-tiff-data/{tiff_id}")
async def clear_tiff_data(
    tiff_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
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
    result_zip_files = glob(f"{result_folder}/{tiff_id}.zip")

    all_files = (
        tiff_files + cell_mask_files + result_files + bg_mask_files + annotation_files + result_zip_files
    )

    logger.info(
        f"Deleting {len(all_files)} files for tiff id: {tiff_id}, from user: {current_user.username}"
    )
    logger.info(f"Files: {all_files}")

    if all_files:
        for file in all_files:
            os.remove(file)

    delete_user_tiff_mappings(db, current_user.username, tiff_id)

    return JSONResponse(
        content={"message": "Tiff data cleared successfully"},
        status_code=200,
    )
