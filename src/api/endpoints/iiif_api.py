"""
IIIF Image API 2.0 endpointy: info.json + dlaždice na požiadanie (TIFF v tiff_folder).
Frontend (OpenSeadragon) musí posielať Authorization: Bearer … pri dlaždiciach
(napr. ajaxHeaders / loadTilesWithAjax podľa verzie OSD).
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from schemas.base import User
from services.auth import get_current_user
from services import iiif_service
from services.iiif_cache_warm import warm_iiif_cache_for_tiff_id

logger = logging.getLogger("uvicorn.access")

router = APIRouter()


def _api_root_from_request(request: Request) -> str:
    return iiif_service.compute_iiif_api_root_from_request(request)


@router.post("/iiif/cache/warm/{tiff_id:path}")
def iiif_warm_cache(
    tiff_id: str,
    current_user: User = Depends(get_current_user),
) -> JSONResponse:
    """
    Predgeneruje všetky IIIF dlaždice do iiif_cache (rovnaké requesty ako OpenSeadragon).
    Môže trvať dlho pri veľkých TIFF; volaj napr. z curl alebo po importe cez Celery.
    """
    try:
        result = warm_iiif_cache_for_tiff_id(tiff_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail="Invalid path") from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return JSONResponse(content=result)


@router.get("/iiif/{tiff_id:path}/info.json")
async def iiif_info(
    request: Request,
    tiff_id: str,
    current_user: User = Depends(get_current_user),
) -> JSONResponse:
    """IIIF info.json – rozmery, tiles, @id pre OpenSeadragon."""
    try:
        info = iiif_service.build_info_json(tiff_id, _api_root_from_request(request))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail="Invalid path") from e
    return JSONResponse(content=info)


@router.get("/iiif/{tiff_id:path}/{region}/{size}/{rotation}/{quality_ext}")
async def iiif_image(
    tiff_id: str,
    region: str,
    size: str,
    rotation: str,
    quality_ext: str,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    IIIF image request: region / size / rotation / quality.fmt
    Príklad: .../full/full/0/default.jpg
    """
    if "." not in quality_ext:
        raise HTTPException(status_code=400, detail="Expected quality.ext e.g. default.jpg")
    quality, fmt = quality_ext.rsplit(".", 1)

    try:
        path = iiif_service.resolve_tiff_path(tiff_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail="Invalid path") from e

    try:
        body = iiif_service.render_iiif_region(
            path, region, size, rotation, quality, fmt, use_cache=True
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    mime = iiif_service.get_mime_for_format(fmt)
    return Response(content=body, media_type=mime)
