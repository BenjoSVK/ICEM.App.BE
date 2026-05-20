import zipfile
from glob import glob
from pathlib import Path

from backend.storage import Storage


def build_result_bundle(storage: Storage, tiff_path: Path) -> Path:
    """Create or refresh a ready-to-download ZIP with WSI and related GeoJSON files."""
    stem = tiff_path.stem
    result_dir = storage.get_folderpath(Storage.RESULT_FOLDER)
    result_path = result_dir / f"{stem}.zip"

    annotation_folder = storage.get_folderpath(Storage.ANNOTATION_FOLDER)
    result_folder = storage.get_folderpath(Storage.RESULT_FOLDER)
    geojson_candidates = []
    geojson_candidates.extend(glob(f"{annotation_folder.as_posix()}/**/*{stem}*.geojson", recursive=True))
    geojson_candidates.extend(glob(f"{result_folder.as_posix()}/**/*{stem}*.geojson", recursive=True))
    geojson_paths = [Path(p).resolve() for p in sorted(set(geojson_candidates))]
    latest_geojson = max(geojson_paths, key=lambda p: p.stat().st_mtime) if geojson_paths else None

    with zipfile.ZipFile(result_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tiff_path, arcname=f"wsi/{tiff_path.name}")
        if latest_geojson is not None:
            zf.write(latest_geojson, arcname=f"geojson/{latest_geojson.name}")

    return result_path
