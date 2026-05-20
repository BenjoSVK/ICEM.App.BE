"""
Helpers for per-user TIFF registry in Postgres.
"""
from __future__ import annotations

from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from schemas.file_info import FileInfo


def ensure_registry_tables(db: Session) -> None:
    """Create registry table/indexes if they do not exist."""
    db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS user_tiff_files (
                id BIGSERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                tiff_id TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                size_bytes BIGINT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'low',
                status TEXT NOT NULL DEFAULT 'ready',
                task_id TEXT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (username, tiff_id)
            );
            """
        )
    )
    db.execute(
        text("ALTER TABLE user_tiff_files ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ready';")
    )
    db.execute(
        text("ALTER TABLE user_tiff_files ADD COLUMN IF NOT EXISTS task_id TEXT NULL;")
    )
    db.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_user_tiff_files_username ON user_tiff_files (username);"
        )
    )
    db.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_user_tiff_files_tiff_id ON user_tiff_files (tiff_id);"
        )
    )
    # Keep a single owner per TIFF id (legacy cleanup for previously duplicated mappings).
    db.execute(
        text(
            """
            DELETE FROM user_tiff_files a
            USING user_tiff_files b
            WHERE a.tiff_id = b.tiff_id
              AND a.id > b.id;
            """
        )
    )
    db.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_user_tiff_files_tiff_id
            ON user_tiff_files (tiff_id);
            """
        )
    )
    db.commit()


def upsert_user_tiff_files(db: Session, username: str, files: Iterable[FileInfo]) -> None:
    """Insert/update per-user TIFF registry rows."""
    for file_info in files:
        db.execute(
            text(
                """
                INSERT INTO user_tiff_files (username, tiff_id, last_modified, size_bytes)
                VALUES (:username, :tiff_id, :last_modified, :size_bytes)
                ON CONFLICT (tiff_id)
                DO UPDATE SET
                    username = EXCLUDED.username,
                    last_modified = EXCLUDED.last_modified,
                    size_bytes = EXCLUDED.size_bytes,
                    updated_at = NOW();
                """
            ),
            {
                "username": username,
                "tiff_id": file_info.id,
                "last_modified": file_info.last_modified,
                "size_bytes": int(file_info.size_bytes),
            },
        )
    db.commit()


def list_user_tiff_files(db: Session, username: str) -> list[dict]:
    rows = db.execute(
        text(
            """
            SELECT tiff_id, last_modified, size_bytes, priority, status, task_id
            FROM user_tiff_files
            WHERE username = :username
            ORDER BY updated_at DESC;
            """
        ),
        {"username": username},
    ).fetchall()
    return [
        {
            "id": row[0],
            "last_modified": row[1],
            "size_bytes": int(row[2]),
            "priority": row[3],
            "status": row[4],
            "task_id": row[5],
        }
        for row in rows
    ]


def list_user_tiff_ids(db: Session, username: str) -> set[str]:
    rows = db.execute(
        text("SELECT tiff_id FROM user_tiff_files WHERE username = :username"),
        {"username": username},
    ).fetchall()
    return {row[0] for row in rows}


def user_owns_tiff(db: Session, username: str, tiff_id: str) -> bool:
    row = db.execute(
        text(
            """
            SELECT 1
            FROM user_tiff_files
            WHERE username = :username
              AND tiff_id = :tiff_id
            LIMIT 1;
            """
        ),
        {"username": username, "tiff_id": tiff_id},
    ).fetchone()
    return row is not None


def delete_user_tiff_mappings(db: Session, username: str, tiff_id_stem: str) -> None:
    db.execute(
        text(
            """
            DELETE FROM user_tiff_files
            WHERE username = :username
              AND (tiff_id = :stem OR tiff_id LIKE :prefix);
            """
        ),
        {
            "username": username,
            "stem": tiff_id_stem,
            "prefix": f"{tiff_id_stem}%",
        },
    )
    db.commit()


def mark_tiffs_pending(db: Session, username: str, tiff_ids: Iterable[str], task_id: str) -> None:
    tiff_ids = list(tiff_ids)
    if not tiff_ids:
        return
    db.execute(
        text(
            """
            UPDATE user_tiff_files
            SET status = 'pending',
                task_id = :task_id,
                updated_at = NOW()
            WHERE username = :username
              AND tiff_id = ANY(:tiff_ids);
            """
        ),
        {"username": username, "tiff_ids": tiff_ids, "task_id": task_id},
    )
    db.commit()


def mark_task_terminal(db: Session, task_id: str, status: str) -> None:
    mapped = "complete" if status.lower() == "success" else "error"
    db.execute(
        text(
            """
            UPDATE user_tiff_files
            SET status = :status,
                task_id = NULL,
                updated_at = NOW()
            WHERE task_id = :task_id;
            """
        ),
        {"status": mapped, "task_id": task_id},
    )
    db.commit()
