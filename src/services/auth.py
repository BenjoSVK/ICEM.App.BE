"""
Authentication service: password hashing, JWT creation/verification, and current-user dependency.
"""
import datetime
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import get_settings
from db_handler import get_db
from schemas.base import TokenData, User


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/ikem_api/token")
settings = get_settings()


def hash_password(password: str) -> str:
    """Hash a password using bcrypt. Use this when storing new passwords."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against stored bcrypt hash. Plaintext comparison is not supported."""
    if not hashed_password or not plain_password:
        return False
    # Only accept bcrypt hashes ($2a$, $2b$, $2y$)
    if not (
        hashed_password.startswith("$2")
        and len(hashed_password) > 20
        and hashed_password[3] == "$"
    ):
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def get_user(db: Session, username: str) -> Optional[User]:
    """Load user by username from the database. Returns None if not found."""
    result = db.execute(
        text("SELECT * FROM users WHERE username = :username"), {"username": username}
    )
    user = result.fetchone()

    if user:
        return User(user[0], user[1])

    return None


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Verify username and password; return User if valid, else None."""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(
    data: dict, expires_delta: Optional[datetime.timedelta] = None
) -> str:
    """Create a short-lived JWT access token for the given payload."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.now() + expires_delta
    else:
        expire = datetime.datetime.now() + datetime.timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a long-lived refresh token (stored in httpOnly cookie)."""
    to_encode = data.copy()
    expire = datetime.datetime.now() + datetime.timedelta(
        days=settings.refresh_token_expire_days
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )


def verify_refresh_token(token: str) -> Optional[str]:
    """Verify refresh token and return username if valid, else None."""
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        if payload.get("type") != "refresh":
            return None
        username = payload.get("sub")
        return username if username else None
    except jwt.PyJWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """FastAPI dependency: validate JWT and return the current User or raise 401."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        if payload.get("type") != "access":
            raise credentials_exception
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
