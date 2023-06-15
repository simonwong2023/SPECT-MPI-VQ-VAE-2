import os
import shutil
import re

# Define the paths for the existing folders
existing_ft_train_folder = "/REST FT (Training) 60s path"
existing_ht_train_folder = "/REST HT (Training) 60s path"
existing_ft_test_folder = "/REST FT (Testing) 60s path"
existing_ht_test_folder = "/REST HT (Testing) 60s path"
existing_ft_val_folder = "/REST FT (Validation) 60s path"
existing_ht_val_folder = "/REST HT (Validation) 60s path"

# Define the paths for the new corresponding folders
new_ft_train_folder = "/Rest FT_(Training) path"
new_ht_train_folder = "/Rest HT_(Training) path"
new_ft_test_folder = "/Rest FT_(Testing) path"
new_ht_test_folder = "/Rest HT_(Testing) path"
new_ft_val_folder = "/Rest FT_(Validation) path"
new_ht_val_folder = "/Rest HT_(Validation) path"

# Create the new corresponding folders if they don't exist
if not os.path.exists(new_ft_train_folder):
    os.makedirs(new_ft_train_folder)
if not os.path.exists(new_ht_train_folder):
    os.makedirs(new_ht_train_folder)
if not os.path.exists(new_ft_test_folder):
    os.makedirs(new_ft_test_folder)
if not os.path.exists(new_ht_test_folder):
    os.makedirs(new_ht_test_folder)
if not os.path.exists(new_ft_val_folder):
    os.makedirs(new_ft_val_folder)
if not os.path.exists(new_ht_val_folder):
    os.makedirs(new_ht_val_folder)

def sort_files(file):
    # Use regex to extract the last number
    match = re.search(r'\d+', file[::-1])  # reverse the string and match the last digit
    if match:
        last_number = int(match.group()[::-1])  # Invert the matching result to obtain the last digit
        return last_number
    else:
        return file

def extract_images(source_folder, destination_folder):
    # Create target folder
    os.makedirs(destination_folder, exist_ok=True)

    # Obtain all image subfolders under the source folder, sorted by creation time
    entries = []
    for root, dirs, files in os.walk(source_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            entries.append((os.stat(dir_path).st_ctime, dir_path))

    sorted_entries = sorted(entries, key=lambda e: e[0])

    # Copy image files from the sorted image subfolders to the destination folder
    for _, subfolder_path in sorted_entries:
        for root, dirs, files in os.walk(subfolder_path):
            # Sort files based on the last number
            sorted_files = sorted(files, key=sort_files)

            for file in sorted_files:
                file_path = os.path.join(root, file)
                if file_path.endswith((".jpg", ".jpeg", ".png", ".gif")):
                    destination_path = os.path.join(destination_folder, file)
                    shutil.copy2(file_path, destination_path)

# Extract images from Rest FT (Training) to New Rest FT (Training)
extract_images(existing_ft_train_folder, new_ft_train_folder)

# Extract images from Rest HT (Training) to New Rest HT (Training)
extract_images(existing_ht_train_folder, new_ht_train_folder)

# Extract images from Rest FT (Testing) to New Rest FT (Testing)
extract_images(existing_ft_test_folder, new_ft_test_folder)

# Extract images from Rest HT (Testing) to New Rest HT (Testing)
extract_images(existing_ht_test_folder, new_ht_test_folder)

# Extract images from Rest FT (Validation) to New Rest FT (Validation)
extract_images(existing_ft_val_folder, new_ft_val_folder)

# Extract images from Rest HT (Validation) to New Rest HT (Validation)
extract_images(existing_ht_val_folder, new_ht_val_folder)
