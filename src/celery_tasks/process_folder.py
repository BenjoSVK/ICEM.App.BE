import os
from celery import Celery
import numpy as np
import re
import torch

try:
    from config import get_settings
except Exception as e:
    from src.config import get_settings

try:
    from iedl_segmentation.models.cell_segmentation.resnet_unet import ResNetUnet
except Exception as e:
    from src.iedl_segmentation.models.cell_segmentation.resnet_unet import ResNetUnet

try:
    from iedl_segmentation.cell_postprocessing import performFilters
except Exception as e:
    print("Error in import")
    print(e)
    from src.iedl_segmentation.cell_postprocessing import performFilters

try:
    from iedl_segmentation.utils.create_background_mask import create_bg_mask_otsu
except Exception as e:
    from src.iedl_segmentation.utils.create_background_mask import create_bg_mask_otsu

try:
    from iedl_segmentation.cell_segmentation_prediction import create_cell_mask
except Exception as e:
    from src.iedl_segmentation.cell_segmentation_prediction import create_cell_mask


celery_app = Celery(
    "my_app", broker="redis://redis:6379/0", backend="redis://redis:6379/0"
)


@celery_app.task(name="celery_tasks.process_folder.predict_structure")
def process_tiff_files(details):
    tiff_files = details.get("tiff_files")
    bg_mask_folder = details.get("bg_mask_folder")
    id_list = details.get("id_list")
    tiff_folder = details.get("tiff_folder")
    cell_mask_folder = details.get("cell_mask_folder")

    settings = get_settings()

    cell_model = ResNetUnet(
        im_channels=settings.im_channels,
        mask_channels=settings.mask_channels,
        down_channels=settings.down_channels,
        mid_channels=settings.mid_channels,
        down_sample=settings.down_sample,
        res_net_layers=settings.res_net_layers,
        use_soft_attention=settings.use_soft_attention,
    )
    cell_model.load_state_dict(
        torch.load(settings.cell_model_path, map_location=settings.device)
    )

    #  ========= create background mask =========

    # loop for creating background mask for every tiff file
    for tiff_file in tiff_files:
        bg_mask = create_bg_mask_otsu(tiff_file)

        # with regex, get the number from file, use regex to get the number from the file name
        tiff_id = re.findall(r"\d+", tiff_file)[0]

        with open(f"{bg_mask_folder}/bg_mask_{tiff_id}.npy", "wb") as f:
            np.save(f, bg_mask.astype(np.uint8))

    # ========= create background mask =========

    #  ========= create cell mask =========

    print("==========Creating cell mask==========")

    # create cell masks
    create_cell_mask(
        tiff_folder,
        bg_mask_folder,
        cell_mask_folder,
        cell_model,
        performFilters,
        id_list,
    )
    # ======== create cell mask =========

    # ========= create tissue mask =========

    # TODO: create tissue mask

    # ========= create tissue mask =========

    # ========= create GeoJSON =========

    # TODO: create GeoJSON

    # ========= create GeoJSON =========

    return {"result": "success"}


@celery_app.task(name="celery_tasks.process_folder.unzip_file")
def unzip_file(details):
    zip_path = details.get("zip_path")
    tiff_folder = details.get("tiff_folder")

    # return {"zip_path": zip_path, "tiff_folder": tiff_folder, "res": os.listdir(".")}

    if zip_path is None:
        return {"error": "zip_path not provided"}

    # check if zip file exists
    if not os.path.exists(zip_path):
        return {"error": "zip file does not exist", "zip_path": zip_path}

    # check if zip file is a zip file
    if not zip_path.endswith(".zip"):
        return {"error": "zip file is not a zip file"}

    #  unzip file into tiff_folder, unzip with python zipfile module
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tiff_folder)

    return {"result": "success"}
