import os
from celery import Celery
import logging
import resource
import numpy as np
import re

try:
    from iedl_segmentation.utils.create_background_mask import create_bg_mask_otsu
except ImportError:
    from src.iedl_segmentation.utils.create_background_mask import create_bg_mask_otsu

celery_app = Celery(
    "my_app", broker="redis://redis:6379/0", backend="redis://redis:6379/0"
)


@celery_app.task(name="celery_tasks.process_folder.predict_structure")
def process_tiff_files(details):
    tiff_files = details.get("tiff_files")
    bg_mask_folder = details.get("bg_mask_folder")

    # check if every tiff file exists
    for tiff_file in tiff_files:
        if not os.path.exists(tiff_file):
            return {"error": "tiff file does not exist", "tiff_file": tiff_file}

    for tiff_file in tiff_files:
        bg_mask = create_bg_mask_otsu(tiff_file)

        # with regex, get the number from file, use regex to get the number from the file name
        tiff_id = re.findall(r"\d+", tiff_file)[0]

        with open(f"{bg_mask_folder}/bg_mask_{tiff_id}.npy", "wb") as f:
            np.save(f, bg_mask.astype(np.uint8))
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
