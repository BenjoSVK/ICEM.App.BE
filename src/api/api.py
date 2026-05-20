"""
API router that mounts auth and data endpoints under /ikem_api.
"""
from fastapi import APIRouter

from .endpoints import auth, data_api, iiif_api

api_router = APIRouter()

api_router.include_router(data_api.router, tags=["Data processing API"])
api_router.include_router(iiif_api.router, tags=["IIIF"])
api_router.include_router(auth.router, tags=["Authentication"])
