#!/usr/bin/env python3
"""
CLI: predgenerovanie IIIF cache pre jeden TIFF.

Spustenie z adresára backendu (kde je app.py), napr. v kontajneri:
  python warm_iiif_cache_cli.py --tiff-id 12128_23.tiff

Vyžaduje načítané .env (IEDL_ROOT_DIR, IIIF_TILE_SIZE, …).
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm IIIF tile cache for one TIFF under tiff_folder.")
    parser.add_argument(
        "--tiff-id",
        required=True,
        help="Relatívna cesta k súboru v tiff_folder (ako v get-tiff-files), napr. 12128_23.tiff",
    )
    parser.add_argument("--count-only", action="store_true", help="Len spočítať počet dlaždíc, negenerovať.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from services import iiif_service
    from services.iiif_cache_warm import count_osd_iiif_tiles, warm_iiif_cache_for_tiff_id

    if args.count_only:
        path = iiif_service.resolve_tiff_path(args.tiff_id)
        w, h = iiif_service.get_image_dimensions(path)
        tw = th = iiif_service.get_tile_size()
        n = count_osd_iiif_tiles(w, h, tw, th)
        print(json.dumps({"tiff_id": args.tiff_id, "width": w, "height": h, "tile_count": n}, indent=2))
        return

    result = warm_iiif_cache_for_tiff_id(args.tiff_id)
    print(json.dumps(result, indent=2))
    if result.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
