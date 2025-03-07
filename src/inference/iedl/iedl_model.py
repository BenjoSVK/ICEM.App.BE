
import cv2
import numpy as np
from typing import List, Callable
from pathlib import Path
from skimage import measure
from patchify import patchify
import torchvision.transforms as transforms

from backend.storage import Storage
from inference.inference_engine import IInferenceModel
from inference.exporter import GeoJsonExporter
from pydantic import BaseModel

from tqdm import tqdm
import torch
import geopandas as gpd
import os
from multiprocessing import Pool
from datetime import datetime

import joblib

SHAPE_THRESHOLD = -1
AREA_THRESHOLD = 60
COLOR_THRESHOLD = 11
CIRCULARITY_THRESHOLD = 0.7
MAX_WORKERS = os.cpu_count()

from iedl_segmentation.utils.smooth_tiled_predictions import (
    predict_img_with_smooth_windowing
)
from iedl_segmentation.configuration import configuration

from iedl_segmentation.models.cell_segmentation.resnet_unet import ResNetUnet
from iedl_segmentation.models.structure_segmentation.unet_model import PyramidAttentionUNet
    
from iedl_segmentation.cell_postprocessing import (
    process_batch
)


class IedlModelConfiguration(BaseModel):
    im_channels: int
    mask_channels: int
    down_channels: List[int]
    mid_channels: List[int]
    down_sample: List[bool]
    res_net_layers: int
    use_soft_attention: bool


DEFAULT_IEDL_MODEL_CONFIGURATION = IedlModelConfiguration(
    im_channels=3,
    mask_channels=4,
    down_channels=[64, 128, 256, 512, 1024],
    mid_channels=[1024, 512],
    down_sample=[True, True, True, True],
    res_net_layers=1,
    use_soft_attention=True
)


class IedlTissueConfig:
    def __init__(self):
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



class NumpyToTensor(Callable):
    def __call__(self, x):
        x = x.astype(np.uint8)
        x = np.transpose(x, axes=(2, 0, 1))
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)
        x = (x / 255.0).float()
        return x


class DistanceTransform(Callable):
    def __call__(self, x):
        """Converts <4;H;W> where first 3 channels are RGB and last is cell mask
        Normalized RGB <0;1> and 3 distance maps for each class
        """
        y = np.zeros((6, x.shape[1], x.shape[2]), dtype=np.float32)
        # First 3 channels are RGB converted into LAB
        y[:3] = x[:3].astype(np.float32) / 255.0
        y[:3] = cv2.cvtColor(y[:3].transpose(1,2,0), cv2.COLOR_RGB2LAB).transpose(2,0,1)

        # Next 3 channels are distance transforms for each class
        d1 = ((x[3] == 1) * 1.0).astype(np.uint8)
        d2 = ((x[3] == 2) * 1.0).astype(np.uint8)
        d3 = ((x[3] == 3) * 1.0).astype(np.uint8)
        y[3] = cv2.distanceTransform(d1, cv2.DIST_L2, 3).astype(np.float32)
        y[4] = cv2.distanceTransform(d2, cv2.DIST_L2, 3).astype(np.float32)
        y[5] = cv2.distanceTransform(d3, cv2.DIST_L2, 3).astype(np.float32)
        return y

        

class MultiScale(Callable):
    def __init__(
        self, 
        scalers_path: Path,
        downscaled: bool=True
    ):
        self.downscaled = downscaled            
        self.scalers = []
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/L_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/A_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/B_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/multiclass_dst1_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/multiclass_dst2_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/multiclass_dst3_scaler.joblib"))


    def __call__(self, x):
        if self.downscaled:
            x = x.transpose(1,2,0)
            x = cv2.resize(x, (x.shape[1]//2, x.shape[0]//2))
            x = x.transpose(2,0,1)

        H,W = x.shape[1], x.shape[2]
        x[0] = self.scalers[0].transform(x[0].reshape(-1,1)).reshape(H,W)
        x[1] = self.scalers[1].transform(x[1].reshape(-1,1)).reshape(H,W)
        x[2] = self.scalers[2].transform(x[2].reshape(-1,1)).reshape(H,W)
        x[3] = self.scalers[3].transform(x[3].reshape(-1,1)).reshape(H,W)
        x[4] = self.scalers[4].transform(x[4].reshape(-1,1)).reshape(H,W)
        x[5] = self.scalers[5].transform(x[5].reshape(-1,1)).reshape(H,W)

        # Add batch dimension
        x = np.expand_dims(x, axis=0)
        return x



class IedlModel(IInferenceModel):
    def __init__(
        self,
        models_path: Path,
        model_configuration: IedlModelConfiguration = DEFAULT_IEDL_MODEL_CONFIGURATION
    ):
        self.is_initialized = False
        self.models_path = models_path
        self.exporter = GeoJsonExporter()
        self.device = torch.device("cpu")

        # Our model works in 256x256
        self.patch_size = (256, 256)
        self.model_configuration = model_configuration
        self.tissue_configuration = IedlTissueConfig()
        self.model = None

        # Paths to models
        self.cell_model_path = models_path / "unet_resnet_final_ikem_cell_seg"
        self.tissue_model_path = models_path / "AdditionalData_PyramidAttentionUNet_multiclass_LAB_batchnorm_scaled_BCE+DC.pt"
        self.scalers_path = models_path / "scalers"


    def lazy_initialize(self):
        """Load pytorch models and stuff"""
        if not self.is_initialized:
            # Do just once!
            self.is_initialized = True

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load the model
            self.cell_model = ResNetUnet(
                im_channels=self.model_configuration.im_channels,
                mask_channels=self.model_configuration.mask_channels,
                down_channels=self.model_configuration.down_channels,
                mid_channels=self.model_configuration.mid_channels,
                down_sample=self.model_configuration.down_sample,
                res_net_layers=self.model_configuration.res_net_layers,
                use_soft_attention=self.model_configuration.use_soft_attention,
            )    
            self.cell_model.load_state_dict(
                torch.load(self.cell_model_path, map_location=torch.device("cpu"))
            )
            self.cell_model = self.cell_model.to(self.device)
            self.cell_model.eval()
            self.cell_transform = transforms.Compose([ NumpyToTensor() ])


            self.tissue_model = PyramidAttentionUNet(
                n_channels=3, 
                n_classes=self.tissue_configuration.n_classes
            )
            self.tissue_model.load_state_dict(
                torch.load(self.tissue_model_path, map_location=torch.device("cpu"))
            )
            self.tissue_model = self.tissue_model.to(self.device)
            self.tissue_model.eval()
            self.tissue_transform = transforms.Compose([
                DistanceTransform(),
                MultiScale(self.scalers_path, downscaled=True)
            ])




        # All is well                


    def process_file(
        self, 
        image: np.ndarray,
        image_file_path: Path, 
        storage: Storage
    ):
        """Run the file through our model


            1. Create the background mask using Otsu's thresholding
            2. Create masks for cells
                    - create patches
                    - predict mask for each patch
                    - save mask as NPY
                    - Post-process the final output
                    - Produce geoJSON annotation
            3. Create tissue mask
        
        """

        job_name = image_file_path.stem

        #-----------------------------------------------
        # 1. Create background mask
        mask_background = self._create_background_mask(image)

        cv2.imwrite("iedl_root_dir/temp/image.png", image)
        cv2.imwrite("iedl_root_dir/temp/mask.png", mask_background * 255)

        #-----------------------------------------------
        # 2. Create mask for all cells
        mask_cells = self._create_cell_mask(image, mask_background)
        cv2.imwrite("iedl_root_dir/temp/mask_cells.png", mask_cells * 255)

        i = image.astype(np.uint8).transpose(2,0,1)
        output_cells = np.stack([ i[0], i[1], i[2], mask_cells])
        # Post processing
        output_cells = performFilters(output_cells, tiff_id="")


        #-----------------------------------------------
        # 3. Create tissue mask
        
        mask_tissue = self._create_tissue_mask(output_cells)



        # Export
        exp_filepath = storage.get_filepath(Storage.ANNOTATION_FOLDER, f"{job_name}.geojson", True)
        self.exporter.write_to_file(mask_tissue, exp_filepath)

        pass



    def _create_cell_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:

        result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # 2. Break down into patches
        PH, PW = self.patch_size
        image_patches = patchify(image, (PH, PW, 3), step=PH).astype(np.uint8)
        mask_patches = patchify(mask, (PH, PW), step=PH)

        NUM_PATCHES_ROWS, NUM_PATCHES_COLS = image_patches.shape[:2]
        num_patches = NUM_PATCHES_COLS * NUM_PATCHES_ROWS
        for k in tqdm(range(num_patches), desc="Cell masks", ncols=80):
            i = k // NUM_PATCHES_COLS
            j = k % NUM_PATCHES_COLS

            # Select the patch
            patch_mask = mask_patches[i,j,0]
            patch_image = image_patches[i,j,0]

            # If there is no tissue, skip
            if np.sum(patch_mask) > 0:
                # Compute prediction
                mask_hat = self._infer_single_patch(patch_image)
                result[i*PH:(i+1)*PH, j*PW:(j+1)*PW] = mask_hat

        # Apply background masking ...
        # TODO:

        return result



    
    def _infer_single_patch(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """Run inference on a single patch"""

        # Transform to tensor
        x = self.cell_transform(image)      # <1;C;H;W> float <0;1>
        x = x.to(self.device)

        # Predict
        with torch.no_grad():
            pred = self.cell_model(x)
            pred = pred[0]            
            classes = torch.argmax(pred, dim=0).detach().cpu().numpy()

        return classes

        

    def _create_background_mask(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """Hmm, asi vyrobime masku"""

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply filters for better thresholding
        image_median = cv2.medianBlur(image_gray, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_equalized = clahe.apply(image_median)

        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(
            image_equalized, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY_INV)

        # Set the kernel size
        kernel = np.ones((20, 20), np.uint8)

        # Use morphological closing to fill holes in the black objects
        closed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        separateObjects = False
        if separateObjects:
            # Set the threshold
            binary_mask = closed_image > 0.5
            # Label the connected components in the binary mask
            closed_image, num_labels = measure.label(
                binary_mask, background=0, return_num=True
            )

        closed_image[closed_image == 255] = 1

        return closed_image


    def _create_tissue_mask(
        self,
        output_cells: np.ndarray            # <4; H; W>
    ) -> np.ndarray:
        
        # Prepare for processing
        H, W = output_cells.shape[1], output_cells.shape[2]
        x = self.tissue_transform(output_cells)

        # Prediction on large images
        pred_combined = predict_img_with_smooth_windowing(
            x, 
            window_size=self.patch_size[0],
            subdivisions=2,
            nb_classes=self.tissue_configuration.n_classes,
            mode="multiclass_pyramid",
            pred_func=(
                lambda img_batch_subdiv: torch.sigmoid(
                        self.tissue_model(img_batch_subdiv).view(
                            img_batch_subdiv.size(dim=0), 
                            self.tissue_configuration.n_classes, 
                            self.patch_size[0], self.patch_size[1]
                        )
                    ).float()
            ),
        )

        # Upscale
        pred_combined = cv2.resize(
                    pred_combined.transpose(1, 2, 0),
                    (W, H),
                    interpolation=cv2.INTER_CUBIC,
                ).transpose(2, 0, 1)

        # Convert into float16
        pred_combined = pred_combined.astype(np.float16)
        return pred_combined



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
    print(f"\t --> START - perform cell filter: {tiff_id}, time: {start}")

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

    # Use multiprocessing to parallelize batch processing
    with Pool(processes=MAX_WORKERS) as pool:
        results = pool.map(process_batch, args_list)

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
    print(f"\t --> END - perform cell filter: {tiff_id}, time: {end}")

    return data
