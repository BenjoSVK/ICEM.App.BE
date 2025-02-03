import os
from celery import Celery
import numpy as np
import re
import torch
from glob import glob

from config import get_settings

from iedl_segmentation.models.cell_segmentation.resnet_unet import ResNetUnet
from iedl_segmentation.cell_postprocessing import performFilters
from iedl_segmentation.utils.create_background_mask import create_bg_mask_otsu
from iedl_segmentation.cell_segmentation_prediction import create_cell_mask
from iedl_segmentation.multilabel_prediction import create_tissue_mask
from iedl_segmentation.utils.tissue_segmentation_export import (
    process_structures_predictions,
    process_cell_mask_to_geojson,
)

celery_app = Celery(
    "my_app",
    broker="redis://vgg_histo_redis:6379/0",
    backend="redis://vgg_histo_redis:6379/0",
)


@celery_app.task(name="celery_tasks.process_folder.predict_structure")
def process_tiff_files(details):
    tiff_files = details.get("tiff_files")
    bg_mask_folder = details.get("bg_mask_folder")
    id_list = details.get("id_list")
    tiff_folder = details.get("tiff_folder")
    cell_mask_folder = details.get("cell_mask_folder")
    result_folder = details.get("result_folder")
    annotations_folder = details.get("annotations_folder")

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
        tiff_id = re.findall(r"\d+_\d+", tiff_file)[0]

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

    print("==========Cell mask created==========")

    # ======== create cell mask =========

    # ========= create tissue mask =========

    print("==========Creating tissue mask==========")

    tissue_config = settings.tissue_config

    tissue_model_path = settings.tissue_model_path

    scalers_path = f"{settings.iedl_root_dir}/trained_models/scalers"

    create_tissue_mask(
        config=tissue_config,
        cell_masks_folder=cell_mask_folder,
        final_folder=result_folder,
        model_path=tissue_model_path,
        id_list=id_list,
        scalers_path=scalers_path,
    )

    print("==========Tissue mask created==========")

    # ========= create tissue mask =========

    # ========= create GeoJSON =========
    print("==========Creating GeoJSON==========")
    # loop for creating GeoJSON for every tiff file
    for tiff_file in tiff_files:
        tiff_id = re.findall(r"\d+_\d+", tiff_file)[0]

        pred_path = f"{result_folder}/tissue_mask_{tiff_id}.npy"
        cell_mask_npy = f"{cell_mask_folder}/cell_mask_{tiff_id}.npy"
        bg_path = f"{bg_mask_folder}/bg_mask_{tiff_id}.npy"

        geoj_path_tissue = f"{annotations_folder}/tissue_mask_{tiff_id}.geojson"
        geoj_path_cell = f"{annotations_folder}/cell_mask_{tiff_id}.geojson"

        process_structures_predictions(
            tiff_path=tiff_file,
            vsi_id=tiff_id,
            pred_path=pred_path,
            bg_path=bg_path,
            output_path=geoj_path_tissue,
        )

        process_cell_mask_to_geojson(cell_mask_npy, geoj_path_cell)

    print("==========GeoJSON created==========")

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

    # check if every file has the following name convention <ID>_<ID2>.ome.tif
    # if not, delete the file and return list of files deleted
    files = glob(f"{tiff_folder}/*.tif")

    print(f"Files in tiff folder: {files}")
    deleted_files = []

    for file in files:
        file_name = file.split("/")[-1]
        if not re.match(r"\d+_\d+.ome.tif", file_name):
            os.remove(f"{file}")
            deleted_files.append(file)

    if deleted_files == []:
        return {"result": "success"}

    else:
        return {"result": "success_with_deleted_files", "deleted_files": deleted_files}
