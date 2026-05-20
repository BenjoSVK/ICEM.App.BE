import logging

import gc
import cv2
import numpy as np
from typing import Any, Dict, List, Optional
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

from inference.iedl.transforms import NumpyToTensor, DistanceTransform, MultiScale
from inference.iedl.utils import performFilters

from iedl_segmentation.utils.smooth_tiled_predictions import (
    predict_img_with_smooth_windowing
)

from iedl_segmentation.models.cell_segmentation.resnet_unet import ResNetUnet
from iedl_segmentation.models.cell_segmentation.attention_unet import AttentionUNet
from iedl_segmentation.models.structure_segmentation.unet_model import PyramidAttentionUNet
    

logger = logging.getLogger("uvicorn.access")

CELL_BATCH_SIZE = 16
ATTENTION_FOREGROUND_THRESHOLD = 0.35
CELL_MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "cell-resnet-v1": {
        "arch": "resnet_unet",
        "checkpoint": "unet_resnet_final_ikem_cell_seg",
        "postprocess": "full",
    },
    "cell-attention-v1": {
        "arch": "attention_unet",
        "checkpoint": "attention_unet_cell_seg.pt",
        "postprocess": "light",
    },
}
DEFAULT_CELL_MODEL_ID = "cell-resnet-v1"

class IedlModelConfiguration(BaseModel):
    im_channels: int
    mask_channels: int
    down_channels: List[int]
    mid_channels: List[int]
    down_sample: List[bool]
    res_net_layers: int
    use_soft_attention: bool
    cell_model_arch: str
    cell_model_id: Optional[str] = None


DEFAULT_IEDL_MODEL_CONFIGURATION = IedlModelConfiguration(
    im_channels=3,
    mask_channels=4,
    down_channels=[64, 128, 256, 512, 1024],
    mid_channels=[1024, 512],
    down_sample=[True, True, True, True],
    res_net_layers=1,
    use_soft_attention=True,
    cell_model_arch="resnet_unet",
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





class IedlModel(IInferenceModel):
    def __init__(
        self,
        models_path: Path,
        model_configuration: IedlModelConfiguration = DEFAULT_IEDL_MODEL_CONFIGURATION
    ):
        self.is_initialized = False
        self.models_path = models_path
        self.exporter = GeoJsonExporter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Our model works in 256x256
        self.patch_size = (256, 256)
        self.model_configuration = model_configuration
        self.tissue_configuration = IedlTissueConfig()
        self.model = None

        # User-facing registry for selectable cell models.
        # `model_id` is stable and can be exposed to API/UI.
        self.cell_model_registry: Dict[str, Dict[str, str]] = dict(CELL_MODEL_REGISTRY)
        self.default_cell_model_id = DEFAULT_CELL_MODEL_ID
        self.arch_to_model_id = {
            "resnet_unet": "cell-resnet-v1",
            "attention_unet": "cell-attention-v1",
        }

        self.selected_cell_model_id: Optional[str] = None
        self.cell_model_path: Optional[Path] = None
        self.tissue_model_path = models_path / "AdditionalData_PyramidAttentionUNet_multiclass_LAB_batchnorm_scaled_BCE+DC.pt"
        self.scalers_path = models_path / "scalers"

    def _geojson_suffix_for_selected_cell_model(self) -> str:
        model_id = self.selected_cell_model_id or self.default_cell_model_id
        if model_id == "cell-attention-v1":
            return "attention"
        if model_id == "cell-resnet-v1":
            return "resnet"
        # Safe fallback for future model ids.
        return model_id.replace("cell-", "").replace("-v", "_v")

    def _resolve_cell_postprocess_mode(self) -> str:
        model_id = self.selected_cell_model_id or self.default_cell_model_id
        model_spec = self.cell_model_registry.get(model_id, {})
        # Keep backward compatibility for old registry rows.
        return model_spec.get("postprocess", "full")



    def get_available_cell_models(self) -> List[Dict[str, Any]]:
        """Public list of selectable cell models for API/UI consumers."""
        result: List[Dict[str, Any]] = []
        for model_id, spec in sorted(self.cell_model_registry.items()):
            result.append(
                {
                    "id": model_id,
                    "arch": spec["arch"],
                    "checkpoint": spec["checkpoint"],
                    "is_default": model_id == self.default_cell_model_id,
                }
            )
        return result

    @classmethod
    def get_registered_cell_models(cls) -> List[Dict[str, Any]]:
        """Class-level access to model registry (no runtime model init required)."""
        result: List[Dict[str, Any]] = []
        for model_id, spec in sorted(CELL_MODEL_REGISTRY.items()):
            result.append(
                {
                    "id": model_id,
                    "arch": spec["arch"],
                    "checkpoint": spec["checkpoint"],
                    "is_default": model_id == DEFAULT_CELL_MODEL_ID,
                }
            )
        return result

    def set_runtime_options(self, options: Dict[str, Any]) -> None:
        """
        Apply request-scoped model options.
        If selected model changes, force reinitialization.
        """
        requested_model_id = options.get("cell_model_id")
        if not requested_model_id:
            return

        if requested_model_id not in self.cell_model_registry:
            available = ", ".join(sorted(self.cell_model_registry.keys()))
            raise ValueError(
                f"Unsupported cell model id: {requested_model_id}. Available: {available}"
            )

        current_model_id = self.model_configuration.cell_model_id
        if current_model_id != requested_model_id:
            self.model_configuration.cell_model_id = requested_model_id
            self.is_initialized = False
            self.cell_model = None
            self.selected_cell_model_id = None
            self.cell_model_path = None
            logger.info(
                "IedlModel.set_runtime_options - switched cell model id to %s",
                requested_model_id,
            )

    def _resolve_cell_model_selection(self) -> tuple[str, str, Path]:
        requested_model_id = self.model_configuration.cell_model_id
        if requested_model_id:
            model_id = requested_model_id
        else:
            # Backward compatibility with existing arch-based config.
            model_id = self.arch_to_model_id.get(
                self.model_configuration.cell_model_arch.lower(),
                self.default_cell_model_id,
            )

        model_spec = self.cell_model_registry.get(model_id)
        if model_spec is None:
            available_model_ids = ", ".join(sorted(self.cell_model_registry.keys()))
            raise ValueError(
                f"Unsupported cell model id: {model_id}. Available: {available_model_ids}"
            )

        cell_arch = model_spec["arch"].lower()
        checkpoint_path = self.models_path / model_spec["checkpoint"]
        return model_id, cell_arch, checkpoint_path

    def lazy_initialize(self):
        """Load pytorch models and stuff"""
        if not self.is_initialized:

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"IedlModel.lazy_initialize - loading models from {self.models_path.as_posix()}")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Resolve requested model and instantiate the matching architecture.
            model_id, cell_arch, cell_model_path = self._resolve_cell_model_selection()
            if cell_arch == "resnet_unet":
                self.cell_model = ResNetUnet(
                    im_channels=self.model_configuration.im_channels,
                    mask_channels=self.model_configuration.mask_channels,
                    down_channels=self.model_configuration.down_channels,
                    mid_channels=self.model_configuration.mid_channels,
                    down_sample=self.model_configuration.down_sample,
                    res_net_layers=self.model_configuration.res_net_layers,
                    use_soft_attention=self.model_configuration.use_soft_attention,
                )
            elif cell_arch == "attention_unet":
                self.cell_model = AttentionUNet(
                    im_channels=self.model_configuration.im_channels,
                    mask_channels=self.model_configuration.mask_channels,
                )
            else:
                raise ValueError(
                    f"Unsupported cell model architecture: {self.model_configuration.cell_model_arch}"
                )

            self.selected_cell_model_id = model_id
            self.cell_model_path = cell_model_path

            logging.info(
                "IedlModel.lazy_initialize - cell model id: %s, arch: %s, checkpoint: %s",
                self.selected_cell_model_id,
                cell_arch,
                self.cell_model_path.as_posix(),
            )
            self.cell_model.load_state_dict(
                torch.load(
                    self.cell_model_path,
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
            self.cell_model = self.cell_model.to(self.device)
            self.cell_model.eval()
            self.cell_transform = transforms.Compose([ NumpyToTensor() ])


            self.tissue_model = PyramidAttentionUNet(
                n_channels=3, 
                n_classes=self.tissue_configuration.n_classes
            )
            self.tissue_model.load_state_dict(
                torch.load(
                    self.tissue_model_path,
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
            self.tissue_model = self.tissue_model.to(self.device)
            self.tissue_model.eval()
            self.tissue_transform = transforms.Compose([
                DistanceTransform(),
                MultiScale(self.scalers_path, downscaled=True)
            ])

            self.is_initialized = True



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
            3. Create tissue mask
            4. Produce GeoJSON annotations
        
        """

        self.lazy_initialize()

        job_name = image_file_path.stem

        #-----------------------------------------------
        # 1. Create background mask
        logging.debug(f"IedlModel.process_file - {image_file_path.name} - Creating background mask")
        mask_background = self._create_background_mask(image)

        #-----------------------------------------------
        # 2. Create mask for all cells
        logging.debug(f"IedlModel.process_file - {image_file_path.name} - Creating cell mask")
        mask_cells = self._create_cell_mask(image, mask_background)

        i = image.astype(np.uint8).transpose(2,0,1)
        output_cells = np.stack([ i[0], i[1], i[2], mask_cells])
        del i, mask_cells
        gc.collect()
        # Post processing

        postprocess_mode = self._resolve_cell_postprocess_mode()
        if postprocess_mode == "full":
            logging.info(f"IedlModel.process_file - {image_file_path.name} - Filtering cell mask (full)")
            output_cells = performFilters(output_cells, tiff_id=image_file_path.stem)
        else:
            logging.info(
                "IedlModel.process_file - %s - Skipping heavy cell filter (mode=%s)",
                image_file_path.name,
                postprocess_mode,
            )
        logging.info(f"IedlModel.process_file - {image_file_path.name} - Cell filter done, creating tissue mask")

        #-----------------------------------------------
        # 3. Create tissue mask
        mask_tissue = self._create_tissue_mask(output_cells)
        del output_cells
        gc.collect()

        logging.info(f"IedlModel.process_file - {image_file_path.name} - Tissue mask done, exporting GeoJSON")

        # Export
        model_suffix = self._geojson_suffix_for_selected_cell_model()
        exp_filepath = storage.get_filepath(
            Storage.ANNOTATION_FOLDER,
            f"{job_name}_{model_suffix}.geojson",
            True,
        )
        self.exporter.write_to_file(mask_tissue, exp_filepath)

        logging.info(f"IedlModel.process_file - {image_file_path.name} - Done")




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

        patch_buffer = []
        index_buffer = []

        for k in tqdm(
            range(num_patches),
            desc="Cell masks",
            ncols=80,
            dynamic_ncols=True,
        ):
            i = k // NUM_PATCHES_COLS
            j = k % NUM_PATCHES_COLS

            # Select the patch
            patch_mask = mask_patches[i,j,0]
            patch_image = image_patches[i,j,0]

            # If there is no tissue, skip
            if np.sum(patch_mask) == 0 or patch_image.mean() > 220:
                continue

            patch_buffer.append(patch_image)
            index_buffer.append((i, j))

            if len(patch_buffer) == CELL_BATCH_SIZE:
                results = self._infer_batch(patch_buffer)
                for (pi, pj), res in zip(index_buffer, results):
                    result[pi * PH:(pi + 1) * PH, pj * PW:(pj + 1) * PW] = res
                patch_buffer, index_buffer = [], []

        if patch_buffer:
            results = self._infer_batch(patch_buffer) 
            for (pi, pj), res in zip(index_buffer, results):
                result[pi * PH:(pi + 1) * PH, pj * PW:(pj + 1) * PW] = res
        
        # Remove any predicted cells outside tissue/background mask.
        result = np.where(mask > 0, result, 0).astype(np.uint8)

        return result



    
    def _infer_batch(self, patches: list) -> list:
        """Batch inferencia pre viacero patches naraz"""
        tensors = [self.cell_transform(p) for p in patches]
        # cell_transform vracia (1,C,H,W) — squeeze batch dim pred stackom
        tensors = [t.squeeze(0) for t in tensors]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            preds = self.cell_model(batch)
            selected_model = self.selected_cell_model_id or self.default_cell_model_id
            if selected_model == "cell-attention-v1":
                # Force stronger behavioral difference for attention model:
                # use foreground confidence instead of class argmax.
                probs = torch.softmax(preds, dim=1)
                background_prob = probs[:, 0, :, :]
                foreground_mask = (1.0 - background_prob) > ATTENTION_FOREGROUND_THRESHOLD
                # Keep tissue pipeline contract (uint8 cell channel).
                classes = foreground_mask.to(torch.uint8).mul(3).detach().cpu().numpy()
            else:
                classes = torch.argmax(preds, dim=1).detach().cpu().numpy()

        return list(classes)

        

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

        pred_combined = cv2.resize(
            pred_combined.transpose(1, 2, 0),
            (W, H),
            interpolation=cv2.INTER_CUBIC,
        ).transpose(2, 0, 1)

        pred_combined = pred_combined.astype(np.float16)
        return pred_combined