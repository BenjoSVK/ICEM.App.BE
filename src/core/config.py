from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, RedisDsn, field_validator


# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

#     BASE_DIR: Path = Path(__file__).parent.parent

#     BACKEND_CORS_ORIGINS: list[AnyHttpUrl] | AnyHttpUrl = []

#     @field_validator('BACKEND_CORS_ORIGINS', mode='before')
#     @classmethod
#     def assemble_cors_origins(cls, value: str | list[str]) -> list[str]:
#         if isinstance(value, str) and not value.startswith('['):
#             return [i.strip() for i in value.split(",")]
#         elif isinstance(value, list):
#             return value

#         raise ValueError(value)

#     CELERY_BROKER_URL: RedisDsn
#     CELERY_BACKEND_URL: RedisDsn



# settings = Settings()
