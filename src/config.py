"""
This module contains the Setting class with all app configurations.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv


@lru_cache()
def get_settings():
    """
    Return the cached settings object.
    """
    return Settings()


class Settings:
    """
    This class contains all configuration loaded from environment files.
    """

    load_dotenv(".env")

    # FOLDERS
    iedl_root_dir = os.environ.get("IEDL_ROOT_DIR")
    reload = os.environ.get("RELOAD")

    app_name = os.environ.get("app_name")
    log_level = os.environ.get("log_level")

    # uvicorn setup
    port = int(str(os.environ.get("uvicorn_port")))
    host = os.environ.get("uvicorn_host")
    workers = int(str(os.environ.get("uvicorn_workers")))

    # REDIS
    # redis_host = os.environ.get("redis_host")
    # redis_port = os.environ.get("redis_port")

    # redis_uri = f"redis://{redis_host}:{redis_port}"
