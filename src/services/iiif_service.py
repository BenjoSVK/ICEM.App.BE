"""
IIIF Image API 2.0 – generovanie dlaždíc na požiadanie z TIFF (alebo iných obrázkov).

Používa pyvips (libvips) na efektívne čítanie výrezov bez načítania celého obrázka.
Voliteľne ukladá vygenerované dlaždice do iiif_cache_dir (view on demand + cache).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import unquote

from starlette.requests import Request

from config import get_settings

logger = logging.getLogger("uvicorn.access")

try:
    import pyvips

    _HAS_PYVIPS = True
except ImportError:
    _HAS_PYVIPS = False
    pyvips = None


def get_iiif_cache_dir() -> Path:
    """Adresár pre cache dlaždíc (pod iedl_root alebo IIIF_CACHE_DIR)."""
    override = (os.environ.get("IIIF_CACHE_DIR") or "").strip()
    if override:
        return Path(override)
    return Path(get_settings().iedl_root_dir) / "iiif_cache"


def get_tile_size() -> int:
    return int(os.environ.get("IIIF_TILE_SIZE", "256"))


def resolve_tiff_path(tiff_id: str) -> Path:
    """
    Vyrieši cestu k súboru v tiff_folder podľa id (relatívna cesta ako v get-tiff-files).
    """
    safe_id = unquote(tiff_id).replace("\\", "/").lstrip("/")
    base = Path(get_settings().iedl_root_dir) / "tiff_folder"
    path = (base / safe_id).resolve()
    base_resolved = base.resolve()
    try:
        path.relative_to(base_resolved)
    except ValueError as e:
        raise PermissionError("Invalid tiff path") from e
    if not path.is_file():
        raise FileNotFoundError(f"TIFF not found: {safe_id}")
    return path


def _image_dimensions_pyvips(path: Path) -> Tuple[int, int]:
    img = pyvips.Image.new_from_file(str(path), access="sequential")
    return img.width, img.height


def get_image_dimensions(path: Path) -> Tuple[int, int]:
    """Šírka a výška obrázka."""
    if _HAS_PYVIPS:
        return _image_dimensions_pyvips(path)
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _scale_factors_for_image(width: int, height: int, tile: int) -> list[int]:
    """Zoznam scaleFactors pre IIIF info.json (mocniny 2, kým je obrázok väčší ako dlaždica)."""
    m = max(width, height)
    factors: list[int] = []
    f = 1
    while m / f >= tile:
        factors.append(f)
        if f >= 2**20:
            break
        f *= 2
    if not factors:
        factors = [1]
    return factors


def get_scale_factors_for_image(width: int, height: int, tile: int) -> list[int]:
    """Verejný alias pre warm cache / nástroje (rovnaké ako v info.json)."""
    return _scale_factors_for_image(width, height, tile)


def compute_iiif_api_root_from_request(request: Request) -> str:
    """
    Vypočíta /ikem_api root pre @id z požiadavky.
    Nepoužívaj hostname typu vgg_histo_backend v prehliadači – ten funguje len v Docker sieti.
    Prehliadač musí volať API na localhost alebo verejnej doméne; to isté musí byť v @id.
    Ak je za reverse proxy, nastav X-Forwarded-Host (a voliteľne X-Forwarded-Proto).
    """
    explicit = os.environ.get("IIIF_PUBLIC_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")

    forwarded_host = request.headers.get("x-forwarded-host")
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_host:
        host = forwarded_host.split(",")[0].strip()
        scheme = (forwarded_proto or request.url.scheme or "http").split(",")[0].strip()
        return f"{scheme}://{host}".rstrip("/") + "/ikem_api"

    base = str(request.base_url).rstrip("/")
    return f"{base}/ikem_api"


def build_info_json(tiff_id: str, api_root: str) -> dict[str, object]:
    """
    IIIF Image API 2.0 info.json.
    api_root: napr. https://host/ikem_api (bez koncového lomítka)
    """
    path = resolve_tiff_path(tiff_id)
    w, h = get_image_dimensions(path)
    tile = get_tile_size()
    # @id musí byť prefix k image endpointu (bez /info.json)
    identifier = quote_identifier_for_url(tiff_id)
    image_id = f"{api_root}/iiif/{identifier}"
    return {
        "@context": "http://iiif.io/api/image/2/context.json",
        "@id": image_id,
        "protocol": "http://iiif.io/api/image",
        "width": w,
        "height": h,
        "profile": ["http://iiif.io/api/image/2/level2.json"],
        "tiles": [
            {
                "width": tile,
                "height": tile,
                "scaleFactors": _scale_factors_for_image(w, h, tile),
            }
        ],
    }


def quote_identifier_for_url(tiff_id: str) -> str:
    """Bezpečné zakódovanie id do cesty (segmenty)."""
    from urllib.parse import quote

    parts = unquote(tiff_id).replace("\\", "/").split("/")
    return "/".join(quote(p, safe="") for p in parts if p or len(parts) == 1)


_CACHE_FORMAT_VERSION = b"iiif_v2_srgb8"


def _normalize_for_web_export(tile: "pyvips.Image") -> "pyvips.Image":
    """
    Alpha → flatten, farebné priestory → sRGB (ak 3+ kanály), hĺbka → 8-bit uchar pre JPEG/PNG.
    """
    import pyvips

    t = tile
    try:
        has_a = t.hasalpha()
    except AttributeError:
        has_a = False
    if has_a:
        t = t.flatten(background=[255, 255, 255])
    if t.bands >= 3:
        try:
            t = t.colourspace("srgb")
        except pyvips.error.Error:
            logger.debug("IIIF: colourspace srgb skipped", exc_info=True)
    if t.format != "uchar":
        if t.format == "ushort":
            t = t.linear(1.0 / 257.0, 0.0)
        elif t.format == "float":
            t = t * 255.0
        elif t.format == "char":
            t = t + 128.0
        t = t.cast("uchar")
    return t


def _cache_key(
    path: Path,
    region: str,
    size: str,
    rotation: str,
    quality: str,
    fmt: str,
) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_FORMAT_VERSION)
    h.update(str(path.resolve()).encode())
    h.update(region.encode())
    h.update(size.encode())
    h.update(rotation.encode())
    h.update(quality.encode())
    h.update(fmt.encode())
    return h.hexdigest()


def render_iiif_region(
    path: Path,
    region: str,
    size: str,
    rotation: str,
    quality: str,
    fmt: str,
    use_cache: bool = True,
    *,
    preloaded_image: Optional[Any] = None,
) -> bytes:
    """
    Vygeneruje obrázok podľa IIIF Image API 2.0 (zjednodušená podmnožina).
    Podporované: region full|x,y,w,h; size full|w,h|pct:n; rotation 0; quality default|color; fmt jpg|png.

    preloaded_image: ak je nastavený (pyvips.Image), nepoužije sa new_from_file – vhodné pri hromadnom warm cache.
    """
    cache_dir = get_iiif_cache_dir()
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(path, region, size, rotation, quality, fmt)
        sub = cache_dir / key[:2]
        ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
        cached = sub / f"{key}.{ext}"
        if cached.is_file():
            return cached.read_bytes()

    if not _HAS_PYVIPS:
        raise RuntimeError("pyvips is required for IIIF tile rendering; install libvips and pyvips")

    if preloaded_image is not None:
        img = preloaded_image
    else:
        img = pyvips.Image.new_from_file(str(path), access="sequential")
    iw, ih = img.width, img.height

    # Region
    if region == "full":
        left, top, rw, rh = 0, 0, iw, ih
    else:
        m = re.match(r"^(\d+),(\d+),(\d+),(\d+)$", region)
        if not m:
            raise ValueError(f"Unsupported region: {region}")
        left, top, rw, rh = (int(m.group(i)) for i in range(1, 5))
        left = max(0, min(left, iw - 1))
        top = max(0, min(top, ih - 1))
        rw = max(1, min(rw, iw - left))
        rh = max(1, min(rh, ih - top))

    tile = img.crop(left, top, rw, rh)

    # Rotation (MVP: 0 only)
    if rotation not in ("0", "0.0"):
        raise ValueError(f"Unsupported rotation: {rotation}")

    # Size
    if size == "full":
        out_w, out_h = rw, rh
    elif size.startswith("pct:"):
        p = float(size[4:])
        s = p / 100.0
        out_w = max(1, int(rw * s))
        out_h = max(1, int(rh * s))
    else:
        parts = size.split(",")
        if len(parts) != 2:
            raise ValueError(f"Unsupported size: {size}")
        sw, sh = parts[0].strip(), parts[1].strip()
        if sw and sh:
            out_w, out_h = int(sw), int(sh)
        elif sw:
            out_w = int(sw)
            scale = out_w / rw
            out_h = max(1, int(rh * scale))
        elif sh:
            out_h = int(sh)
            scale = out_h / rh
            out_w = max(1, int(rw * scale))
        else:
            out_w, out_h = rw, rh

    # Resize (preserve aspect if only one dimension in pct handled above)
    if out_w != rw or out_h != rh:
        tile = tile.thumbnail_image(out_w, height=out_h, size="force")

    # Quality (default: no extra processing; gray: bandextract if RGB)
    if quality == "gray" and tile.bands >= 3:
        tile = tile.colourspace("b-w")

    tile = _normalize_for_web_export(tile)

    fmt_l = fmt.lower()
    if fmt_l in ("jpg", "jpeg"):
        buf = tile.write_to_buffer(".jpg", Q=85)
        mime = "image/jpeg"
    elif fmt_l == "png":
        buf = tile.write_to_buffer(".png")
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if use_cache:
        key = _cache_key(path, region, size, rotation, quality, fmt)
        sub = cache_dir / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if fmt_l in ("jpg", "jpeg") else "png"
        cached = sub / f"{key}.{ext}"
        try:
            cached.write_bytes(buf)
        except OSError as e:
            logger.warning("IIIF cache write failed: %s", e)

    return buf


def get_mime_for_format(fmt: str) -> str:
    fmt_l = fmt.lower()
    if fmt_l in ("jpg", "jpeg"):
        return "image/jpeg"
    if fmt_l == "png":
        return "image/png"
    return "application/octet-stream"
