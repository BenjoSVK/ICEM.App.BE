"""
Predgenerovanie IIIF dlaždíc do iiif_cache (rovnaké region/size ako OpenSeadragon IIIF v2).

Použitie:
  - warm_iiif_cache_for_tiff_id("foo.tif") — jeden súbor
  - voliteľne po importe ZIP (Celery + IIIF_WARM_ON_IMPORT)
  - POST /ikem_api/iiif/warm-cache/... (manuálne)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterator, Tuple

from services import iiif_service

logger = logging.getLogger(__name__)


def _osd_max_level_from_scale_factors(scale_factors: list[int]) -> int:
    """Rovnaké ako OpenSeadragon: maxLevel = round(log2(max(scaleFactors)))."""
    max_sf = max(scale_factors)
    return int(round(math.log2(max_sf)))


def iter_osd_iiif_tile_requests(
    width: int,
    height: int,
    tile_w: int,
    tile_h: int,
) -> Iterator[Tuple[str, str]]:
    """
    Generuje (region, size) pre všetky dlaždice, ktoré by IIIFTileSource v OSD 4.x
    s IIIF Image API 2.0 a jedným záznamom tiles[].scaleFactors požadoval.
    """
    scale_factors = iiif_service.get_scale_factors_for_image(width, height, tile_w)
    max_level = _osd_max_level_from_scale_factors(scale_factors)

    for level in range(0, max_level + 1):
        scale = 0.5 ** (max_level - level)
        level_width = math.ceil(width * scale)
        level_height = math.ceil(height * scale)
        tile_width = tile_w
        tile_height = tile_h
        iiif_tile_size_w = round(tile_width / scale)
        iiif_tile_size_h = round(tile_height / scale)

        # Malý obrázok / úroveň zmestená do jednej dlaždice (getTileUrl vetva „full“)
        if level_width < tile_width and level_height < tile_height:
            if level_width == width and level_height == height:
                yield ("full", "full")
            else:
                yield ("full", f"{level_width},")
            continue

        num_x = math.ceil(scale * width / tile_width)
        num_y = math.ceil(scale * height / tile_height)

        for x in range(num_x):
            for y in range(num_y):
                iiif_tile_x = x * iiif_tile_size_w
                iiif_tile_y = y * iiif_tile_size_h
                iiif_tile_w = min(iiif_tile_size_w, width - iiif_tile_x)
                iiif_tile_h = min(iiif_tile_size_h, height - iiif_tile_y)

                if x == 0 and y == 0 and iiif_tile_w == width and iiif_tile_h == height:
                    iiif_region = "full"
                else:
                    iiif_region = f"{iiif_tile_x},{iiif_tile_y},{iiif_tile_w},{iiif_tile_h}"

                iiif_size_w = min(tile_width, level_width - (x * tile_width))
                iiif_size_h = min(tile_height, level_height - (y * tile_height))

                # IIIF v2, nie level0: default vetva z iiiftilesource.js
                if iiif_size_w == width:
                    iiif_size = "full"
                else:
                    iiif_size = f"{iiif_size_w},{iiif_size_h}"

                yield (iiif_region, iiif_size)


def count_osd_iiif_tiles(width: int, height: int, tile_w: int, tile_h: int) -> int:
    return sum(1 for _ in iter_osd_iiif_tile_requests(width, height, tile_w, tile_h))


def warm_iiif_cache_for_path(
    path: Path,
    *,
    fmt: str = "jpg",
    quality: str = "default",
    rotation: str = "0",
) -> dict:
    """
    Otvorí TIFF raz, prejde všetky OSD IIIF dlaždice a uloží ich do iiif_cache
    (cez render_iiif_region s preloaded_image).
    """
    if not iiif_service._HAS_PYVIPS:
        raise RuntimeError("pyvips is required for IIIF warm cache")

    import pyvips

    w, h = iiif_service.get_image_dimensions(path)
    tile_w = tile_h = iiif_service.get_tile_size()

    total = count_osd_iiif_tiles(w, h, tile_w, tile_h)
    done = 0
    errors: list[str] = []

    img = pyvips.Image.new_from_file(str(path), access="random")

    for region, size in iter_osd_iiif_tile_requests(w, h, tile_w, tile_h):
        try:
            iiif_service.render_iiif_region(
                path,
                region,
                size,
                rotation,
                quality,
                fmt,
                use_cache=True,
                preloaded_image=img,
            )
            done += 1
            if done % 200 == 0 or done == total:
                logger.info("IIIF warm cache %s: %s / %s tiles", path.name, done, total)
        except Exception as e:
            msg = f"{region}|{size}: {e}"
            errors.append(msg)
            logger.warning("IIIF warm cache tile failed: %s", msg)

    return {
        "path": str(path),
        "width": w,
        "height": h,
        "tiles_total": total,
        "tiles_ok": done,
        "errors": errors,
    }


def warm_iiif_cache_for_tiff_id(tiff_id: str, **kwargs) -> dict:
    """Warm cache pre id relatívne voči tiff_folder (ako get-tiff-files)."""
    path = iiif_service.resolve_tiff_path(tiff_id)
    return warm_iiif_cache_for_path(path, **kwargs)
