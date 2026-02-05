import jwt
import datetime
import bcrypt

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from sqlalchemy import text
from sqlalchemy.orm import Session


from db_handler import get_db
from schemas.base import User, TokenData
from config import get_settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/ikem_api/token")
settings = get_settings()


def hash_password(password: str) -> str:
    """Hash a password using bcrypt. Use this when storing new passwords."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, db_password: str) -> bool:
    """Verify password against stored hash. Supports bcrypt hashes and plaintext (legacy)."""
    if not db_password:
        return False
    if db_password.startswith("$2") and len(db_password) > 20:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), db_password.encode("utf-8")
        )
    return plain_password == db_password


def get_user(db, username: str):
    result = db.execute(
        text("SELECT * FROM users WHERE username = :username"), {"username": username}
    )
    user = result.fetchone()

    if user:
        return User(user[0], user[1])

    return None


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.now() + expires_delta
    else:
        expire = datetime.datetime.now() + datetime.timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
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
