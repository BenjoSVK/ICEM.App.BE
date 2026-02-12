"""
Authentication endpoints: login (token + refresh cookie), refresh, logout.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

logger = logging.getLogger("uvicorn.access")

from api.limiter import limiter
from config import get_settings
from db_handler import get_db
from services.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
)

router = APIRouter()
settings = get_settings()

REFRESH_TOKEN_COOKIE = "refresh_token"

# Rate limit: 5 login attempts per 15 minutes per IP (brute-force protection)
LOGIN_RATE_LIMIT = "10/15 minutes"


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    """Set httpOnly cookie with refresh token."""
    max_age = settings.refresh_token_expire_days * 24 * 60 * 60
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE,
        value=refresh_token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=settings.is_prod,
        path="/",
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Clear refresh token cookie (logout)."""
    response.delete_cookie(
        key=REFRESH_TOKEN_COOKIE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.is_prod,
    )


@router.post("/token")
@limiter.limit(LOGIN_RATE_LIMIT)
async def login_for_access_token(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> dict:
    """Issue access token and set refresh token in httpOnly cookie."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    _set_refresh_cookie(response, refresh_token)
    logger.info(f"User {user.username} logged in")
    expires_seconds = settings.access_token_expire_minutes * 60
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_seconds,
    }


@router.post("/refresh")
async def refresh_access_token(
    request: Request, response: Response
) -> dict:
    """Issue a new access token using the refresh token from httpOnly cookie."""
    refresh = request.cookies.get(REFRESH_TOKEN_COOKIE)
    if not refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username = verify_refresh_token(refresh)
    if not username:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": username})
    expires_seconds = settings.access_token_expire_minutes * 60
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_seconds,
    }


@router.post("/logout")
async def logout(response: Response) -> dict:
    """Clear refresh token cookie."""
    _clear_refresh_cookie(response)
    return {"detail": "Logged out"}
