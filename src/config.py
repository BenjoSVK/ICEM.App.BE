"""
This module contains the Setting class with all app configurations.
"""

import os
from functools import lru_cache
import torch

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

    app_name = os.environ.get("APP_NAME")
    log_level = os.environ.get("LOG_LEVEL")

    # uvicorn setup
    port = int(str(os.environ.get("UVICORN_PORT")))
    host = os.environ.get("UVICORN_HOST")
    workers = int(str(os.environ.get("UVICORN_WORKERS")))

    pg_port = int(str(os.environ.get("PG_PORT")))
    pg_user = os.environ.get("PG_USER")
    pg_password = os.environ.get("PG_PASSWORD")
    pg_host = os.environ.get("PG_HOST")
    pg_database = os.environ.get("PG_DB")

    db_uri = (
        "postgresql://"
        f"{pg_user}"
        ":"
        f"{pg_password}"
        "@"
        f"{pg_host}"
        ":"
        f"{pg_port}"
        "/"
        f"{pg_database}"
    )

    secret_key = os.environ.get("SECRET_KEY")
    algorithm = os.environ.get("ALGORITHM")
    access_token_expire_minutes = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES"))

    # setup for models
    im_channels = 3
    mask_channels = 4
    down_channels = [64, 128, 256, 512, 1024]
    mid_channels = [1024, 512]
    down_sample = [True, True, True, True]
    res_net_layers = 1
    use_soft_attention = True

    cell_model_path = "/iedl_root_dir/trained_models/unet_resnet_final_ikem_cell_seg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
