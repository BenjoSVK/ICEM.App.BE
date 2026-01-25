import geopandas as gpd
import numpy as np
from pathlib import Path
import cv2
from shapely.geometry import shape
from tqdm import tqdm

from iedl_segmentation.utils.tissue_segmentation_export import get_name_and_color_tissue


class GeoJsonExporter:
    def __init__(self):

        # Do we need this ?
        
        pass




    def write_to_file(
        self, 
        tissue_mask: np.ndarray,
        filepath: Path
    ):
        """Converts the tissue mask into a JSON file with polygon annotations"""

        # We use the original implementation from Matej Halinkovic
        _convert_to_geojson(
            tissue_mask=tissue_mask,
            output_path=filepath.as_posix(),
            vsi_id=""
        )


"""
    This is changed - the original implementation did not handle correctly
    the case when no contours were detected.
"""


def _convert_to_geojson(tissue_mask: np.ndarray, output_path: str, vsi_id: str):
    """
    Converts a numpy array mask into a GeoJSON file with polygon annotations.
    Args:
        npyData (str): Path to the numpy array file containing the mask data.
        outputPath (str): Directory where the output GeoJSON file will be saved.
        vsiId (str): Identifier for the VSI (Virtual Slide Image).
    Returns:
        None
    The function processes the mask data to extract contours for different classes,
    converts them into polygons, and saves them as a GeoJSON file with appropriate
    metadata for each polygon.
    """

    geojson_cells_classes = []

    for c_idx, thresh, color in zip(
        [1, 2, 3], 
        [0.15, 0.15, 0.5], 
        [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
    ):

        contours, _ = cv2.findContours(
            (tissue_mask[c_idx] >= thresh).astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in tqdm(contours):

            if len(contour) < 3:
                continue  # Skip if not enough points for a polygon

            cell_class = c_idx

            # Convert contour to the correct format
            contour = np.squeeze(contour)
            if len(contour.shape) != 2 or contour.shape[1] != 2:
                continue  # Skip if the contour is not valid after squeezing

            # Save class to GeoJSON structure
            contour = np.squeeze(contour)
            polygon_coords = [contour.tolist()]
            polygon = shape({"type": "Polygon", "coordinates": polygon_coords})

            name, color = get_name_and_color_tissue(cell_class - 1)
            meta_data_cell_img = {"name": name, "color": color}

            # Create GeoJSON feature with properties
            predicted_cell_class = {
                "type": "Feature",
                "geometry": polygon,
                "properties": {
                    "objectType": "annotation",
                    "classification": meta_data_cell_img,
                },
            }

            geojson_cells_classes.append(predicted_cell_class)

    if len(geojson_cells_classes) > 0:
        gdf_classes = gpd.GeoDataFrame.from_features(geojson_cells_classes)
        print(f"Saving GeoJSON file to {output_path}")
        gdf_classes.to_file(f"{output_path}", driver="GeoJSON")
    else:
        print(f"No contours detected. Skipping writing GeoJSON file.")