from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from fastapi import APIRouter
import logging

logger = logging.getLogger("uvicorn.access")

from api.limiter import limiter
from db_handler import get_db
from services.auth import authenticate_user, create_access_token

router = APIRouter()

# Rate limit: 5 login attempts per 15 minutes per IP (brute-force protection)
LOGIN_RATE_LIMIT = "10/15 minutes"


@router.post("/token")
@limiter.limit(LOGIN_RATE_LIMIT)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    logger.info(f"User {user.username} logged in")
    return {"access_token": access_token, "token_type": "bearer"}
