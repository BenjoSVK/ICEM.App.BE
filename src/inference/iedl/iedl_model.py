
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
from iedl_segmentation.models.cell_segmentation.resnet_unet import ResNetUnet


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


class NumpyToTensor(Callable):
    def __call__(self, x):
        x = x.astype(np.uint8)
        x = np.transpose(x, axes=(2, 0, 1))
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)
        x = (x / 255.0).float()
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
        self.model = None

        # Paths to models
        self.cell_model_path = models_path / "unet_resnet_final_ikem_cell_seg"
        self.tissue_model_path = models_path / "AdditionalData_PyramidAttentionUNet_multiclass_LAB_batchnorm_scaled_BCE+DC.pt"


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

        
        """

        job_name = image_file_path.stem

        #-----------------------------------------------
        # 1. Create background mask
        mask_background = self._create_background_mask(image)

        #-----------------------------------------------
        # 2. Create mask for all cells
        mask_cells = self._create_cell_mask(image, mask_background)
        i = image.astype(np.uint8).transpose(2,0,1)
        output_cells = np.stack([ i[0], i[1], i[2], mask_cells])
        mask_filepath = storage.get_filepath(Storage.CELL_MASK_FOLDER, 
            f"cell_mask_{job_name}.npy", create_parents=True
        )
        np.save(mask_filepath, output_cells)

        exp_filepath = storage.get_filepath(Storage.ANNOTATION_FOLDER,     
            f"{job_name}.geojson", create_parents=True
        )
        self.exporter.write_to_file(mask_cells, exp_filepath)

        


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
