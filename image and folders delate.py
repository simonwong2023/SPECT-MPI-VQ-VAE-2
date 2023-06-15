import os
import shutil

def remove_subfolders_and_images(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            shutil.rmtree(dir_path)

# List of folders to remove subfolders and images from
folders = [
    "/Rest FT_(Training) path",
    "/Rest HT_(Training) path",
    "/Rest FT_(Testing) path",
    "Rest HT_(Testing) path",
    "/Rest FT_(Validation) path",
    "/Rest HT_(Validation) path"
]

# Remove subfolders and images from each folder
for folder in folders:
    remove_subfolders_and_images(folder)


