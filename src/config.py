"""
This module contains the Setting class with all app configurations.
"""

import os
from functools import lru_cache
from typing import Optional

import torch
from dotenv import load_dotenv

_SECRET_KEY_PLACEHOLDER = "your_secret_key_here"


def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get env var; return default or raise ValueError if required and missing."""
    value = os.environ.get(key)
    if value is not None and value.strip() != "":
        return value.strip()
    if required:
        raise ValueError(
            f"Missing required environment variable: {key}. Set it in .env or export {key}=<value>."
        )
    return default


def _get_env_int(key: str, default: Optional[int] = None, required: bool = False) -> int:
    """Get env var as int; return default or raise ValueError if invalid/missing."""
    value = os.environ.get(key)
    if value is not None and value.strip() != "":
        try:
            return int(value.strip())
        except ValueError as e:
            raise ValueError(f"Environment variable {key} must be an integer, got: {value!r}") from e
    if required:
        raise ValueError(
            f"Missing required environment variable: {key}. Set it in .env or export {key}=<integer>."
        )
    if default is None:
        raise ValueError(f"Environment variable {key} is missing and no default was provided.")
    return default


def _validate_secret_key(value: Optional[str], is_prod: bool) -> str:
    """Ensure SECRET_KEY is set; in prod reject placeholder. In dev allow placeholder → use dev default."""
    raw = (value or "").strip()
    if is_prod:
        if not raw:
            raise ValueError(
                "SECRET_KEY is not set. Set SECRET_KEY in .env (e.g. python -c \"import secrets; print(secrets.token_hex(32))\")."
            )
        if raw == _SECRET_KEY_PLACEHOLDER:
            raise ValueError(
                "SECRET_KEY must be changed from the placeholder 'your_secret_key_here'. Set a secure key in .env."
            )
        return raw
    if not raw or raw == _SECRET_KEY_PLACEHOLDER:
        return "dev-secret-key-not-for-production"
    return raw


@lru_cache()
def get_settings() -> "Settings":
    """
    Return the cached settings object.
    """
    return Settings()


class Settings:
    """
    Configuration from .env. Uses safe getters; raises ValueError on missing required vars or invalid types.
    """

    load_dotenv(".env")

    deploy = (_get_env("DEPLOY", default="dev") or "dev").strip().lower()
    is_prod = deploy == "prod"

    # FOLDERS
    iedl_root_dir = _get_env("IEDL_ROOT_DIR", default="/iedl_root_dir")
    reload = _get_env("RELOAD", default="False")

    app_name = _get_env("APP_NAME", default="iedl-api")
    log_level = _get_env("LOG_LEVEL", default="INFO" if is_prod else "DEBUG")

    # Redis
    redis_host = _get_env("REDIS_HOST", default="localhost")
    redis_port = _get_env_int("REDIS_PORT", default=6379)

    # uvicorn setup
    port = _get_env_int("UVICORN_PORT", default=8000)
    host = _get_env("UVICORN_HOST", default="0.0.0.0") or "0.0.0.0"
    workers = _get_env_int("UVICORN_WORKERS", default=1)

    pg_port = _get_env_int("PG_PORT", default=5432)
    pg_user = _get_env("PG_USER", default="postgres") or "postgres"
    pg_password = _get_env("PG_PASSWORD", default="")
    pg_host = _get_env("PG_HOST", default="localhost") or "localhost"
    pg_database = _get_env("PG_DB", default="mydatabase") or "mydatabase"

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

    _cors_raw = _get_env("CORS_ORIGINS", default="*" if not is_prod else "")
    cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()] if _cors_raw.strip() else (["*"] if not is_prod else [])

    secret_key = _validate_secret_key(_get_env("SECRET_KEY", default=""), is_prod)
    algorithm = _get_env("ALGORITHM", default="HS256") or "HS256"
    access_token_expire_minutes = _get_env_int("ACCESS_TOKEN_EXPIRE_MINUTES", default=15)
    refresh_token_expire_days = _get_env_int("REFRESH_TOKEN_EXPIRE_DAYS", default=7)

    # setup for models
    im_channels = 3
    mask_channels = 4
    down_channels = [64, 128, 256, 512, 1024]
    mid_channels = [1024, 512]
    down_sample = [True, True, True, True]
    res_net_layers = 1
    use_soft_attention = True

    cell_model_path = "/iedl_root_dir/trained_models/unet_resnet_final_ikem_cell_seg"
    tissue_model_path = "/iedl_root_dir/trained_models/AdditionalData_PyramidAttentionUNet_multiclass_LAB_batchnorm_scaled_BCE+DC.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    class TissueConfig:
        """Configuration for tissue segmentation model (LR, epochs, mode, etc.)."""

        def __init__(self) -> None:
            self.log = False
            self.lr = 5e-6
            self.n_classes = 5
            self.l2_lambda = 1e-6
            self.batch_size = 16
            self.epochs = 25
            self.mode = "multiclass_pyramid"
            self.norm = "batch"
            self.nodst = False
            self.colorspace = "lab"
            self.attention = True

    # Replace the dictionary with an instance of the new class
    tissue_config = TissueConfig()
