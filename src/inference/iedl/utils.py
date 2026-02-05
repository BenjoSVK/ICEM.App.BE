
import os
import logging
import cv2
import numpy as np
import geopandas as gpd

from datetime import datetime
from multiprocessing import Pool


from iedl_segmentation.cell_postprocessing import (
    process_batch
)

logger = logging.getLogger(__name__)


SHAPE_THRESHOLD = -1
AREA_THRESHOLD = 60
COLOR_THRESHOLD = 11
CIRCULARITY_THRESHOLD = 0.7
MAX_WORKERS = 1 #os.cpu_count()




"""
    Modified post-processing
"""

def performFilters(
    data: np.array,
    area_threshold: int = AREA_THRESHOLD,
    shape_threshold: int = SHAPE_THRESHOLD,
    circularity_threshold: int = CIRCULARITY_THRESHOLD,
    color_threshold: int = COLOR_THRESHOLD,
    output_path: str = None,
    create_geojson: bool = False,
    batch_size_set: int = 0,
    tiff_id: str = None,
) -> np.array:

    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.warning(f"START - perform cell filter: tiff_id={tiff_id}, time={start}")

    geojson_cells_classes = []

    # Prepare whole_image and filtered_masks
    whole_image = data[:3, :, :].transpose(1, 2, 0)
    filtered_masks = data[3, :, :].astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        filtered_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    new_mask = np.zeros_like(filtered_masks)

    total_contours = len(contours)
    batch_size = total_contours // MAX_WORKERS + (total_contours % MAX_WORKERS > 0)

    if batch_size_set != 0:
        batch_size = batch_size_set
    elif MAX_WORKERS == 1 and total_contours > 500:
        # Process in chunks so we can log progress (avoids "stuck" appearance)
        batch_size = 500

    # Prepare arguments for parallel processing
    args_list = []
    for i in range(0, total_contours, batch_size):
        batch_contours = contours[i : min(i + batch_size, total_contours)]
        args = (
            whole_image,
            filtered_masks,
            batch_contours,
            area_threshold,
            shape_threshold,
            circularity_threshold,
            color_threshold,
            create_geojson,
        )
        args_list.append(args)

    num_batches = len(args_list)
    logger.warning(f"cell filter: total_contours={total_contours}, num_batches={num_batches}, batch_size={batch_size}")
    # Use multiprocessing to parallelize batch processing
    if MAX_WORKERS > 1:
        with Pool(processes=MAX_WORKERS) as pool:
            results = pool.map(process_batch, args_list)
    else:
        results = []
        for idx, x in enumerate(args_list):
            results.append(process_batch(x))
            logger.warning(f"cell filter progress: batch {idx + 1}/{num_batches}, time: {datetime.now().strftime('%H:%M:%S')}")

    # Combine results
    for result in results:
        contour_mask = result["contour_mask"]
        new_mask = cv2.bitwise_or(new_mask, contour_mask)

        if create_geojson:
            geojson_cells_classes.extend(result["geojson_cells"])

    # Save mask and GeoJSON if required
    if output_path:
        with open(f"{output_path}/filtered_mask.npy", "wb") as f:
            np.save(f, new_mask.astype(np.uint8))

    if create_geojson:
        gdf_classes = gpd.GeoDataFrame.from_features(geojson_cells_classes)
        gdf_classes.to_file(f"{output_path}/filtered_mask.geojson", driver="GeoJSON")

    # Update data with new mask
    data[3, :, :] = new_mask

    end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.warning(f"END - perform cell filter: tiff_id={tiff_id}, time={end}")

    return data
