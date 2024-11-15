import os

# from core.celery import celery_app
from celery import Celery


celery_app = Celery(
    "my_app", broker="redis://redis:6379/0", backend="redis://redis:6379/0"
)


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
