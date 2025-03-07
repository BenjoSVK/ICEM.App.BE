import argparse

from pathlib import Path
import shutil

from backend.storage import Storage
from backend.inference_backend import InferenceBackend
from backend.file_handlers import FileHandlers
from backend.handlers.zip_handler import ZipArchiveHandler
from backend.handlers.image_handler import ImageFileHandler

def main(args):


    backend = InferenceBackend(
        handlers=FileHandlers([
            ZipArchiveHandler(),
            ImageFileHandler()
        ]),
        storage=Storage(basepath=Path(args.storage))
    )

    #1 - Copy the given file to the ZIP folder    
    target_filename = backend.storage.get_filepath(
        Storage.ZIP_FOLDER, 
        Path(args.input_file).name,
        create_parents=True
    )
    print(f"Copying {args.input_file} -> {target_filename.as_posix()}")
    shutil.copyfile(args.input_file, target_filename)

    #2 - Accept the file by the backend
    accepted_files = backend.accept_file(target_filename)
    if len(accepted_files) == 0:
        print("Error! - no valid files were extracted from the archive")
    else:
        print("Accepted files: ")
        for file_info in accepted_files:
            print(f"   {file_info}")

    #3 - List what we've got
    files = backend.get_available_inference_files()
    print("")
    print(f"Available files: ")
    for file_info in files:
        print(f"   {file_info}")





if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_file", "-i", 
        help="ZIP file to be uploaded to the service"
        )
    p.add_argument(
        "--storage", "-s", 
        default="./iedl_root_dir", 
        help="Base folder for storing backend files"
        )

    main(p.parse_args())



