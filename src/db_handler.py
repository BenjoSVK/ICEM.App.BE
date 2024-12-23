"""
This module handles database initiation and the creation/destruction of DB connection.
"""

from sqlalchemy.engine import Engine

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import get_settings


base = object
Session_Local = Session
deimos_db = Engine


def create_database_connection() -> None:
    """
    Init function for creating connection with database.
    Needs to be called when app is created. Evaluates env variables and
    if testing is enabled, it returns mocked `base` and `db_session`.
    """
    global deimos_db  # for raw sql calls
    global base
    global Session_Local

    deimos_db = create_engine(
        get_settings().db_uri,
        pool_recycle=3600,
        connect_args={"options": "-c timezone=Europe/Bratislava"},
    )

    Session_Local = sessionmaker(autocommit=False, autoflush=False, bind=deimos_db)


def get_session() -> Session:
    """
    Retrieve the session object
    """
    return Session_Local()


def get_db():
    """
    Create DB connection for API request and close it after request is done
    """
    db = Session_Local()
    try:
        yield db
    finally:
        db.close()


create_database_connection()
