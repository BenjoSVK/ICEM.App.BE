import geopandas as gpd
import numpy as np
from pathlib import Path


from iedl_segmentation.utils.tissue_segmentation_export import convert_to_geojson


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
        convert_to_geojson(
            tissue_mask=tissue_mask,
            output_path=filepath.as_posix(),
            vsi_id=""
        )
