# Split subfolders from two original directories into three new directories for
# training sets (FT & HT), testing sets (FT & HT), and validation sets (FT & HT) randomly.

import os
import random
import shutil
import  time

# Define the paths for the original folders and the new folders
original_ft_folder = "/REST FT path"
original_ht_folder = "/REST HT path"

new_ft_train_folder = "/REST FT (Training) 60s path"
new_ht_train_folder = "/REST HT (Training) 60s path"
new_ft_test_folder = "/REST FT (Testing) 60s path"
new_ht_test_folder = "/REST HT (Testing) 60s path"
new_ft_val_folder = "/REST FT (Validation) 60s path"
new_ht_val_folder = "/REST HT (Validation) 60s path"

# Create the new folders if they don't exist
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

# Get a list of subfolders in the original FT folder
ft_subfolders = os.listdir(original_ft_folder)
ft_subfolders = sorted(ft_subfolders)  # Sort the subfolders to ensure order

# Get a list of subfolders in the original HT folder
ht_subfolders = os.listdir(original_ht_folder)
ht_subfolders = sorted(ht_subfolders)  # Sort the subfolders to ensure order

# Calculate the number of subfolders for each set
num_subfolders = len(ft_subfolders)
num_train_subfolders = int(0.8 * num_subfolders) # 480
num_test_subfolders = int(0.1 * num_subfolders) # 60 
num_val_subfolders = int(0.1 * num_subfolders) # 60

# Randomly select the subfolders for each set
selected_subfolders = random.sample(range(num_subfolders), num_subfolders)
random.shuffle(selected_subfolders)

# Obtain subfolders index for each collection
train_subfolders = selected_subfolders[:num_train_subfolders]
test_subfolders = selected_subfolders[num_train_subfolders:num_train_subfolders + num_test_subfolders]
val_subfolders = selected_subfolders[-num_test_subfolders:]


# Copy and paste subfolders for the training set
for index in train_subfolders:
    subfolder_name_ft = ft_subfolders[index]
    subfolder_name_ht = ht_subfolders[index]

    source_ft = os.path.join(original_ft_folder, subfolder_name_ft)
    source_ht = os.path.join(original_ht_folder, subfolder_name_ht)

    if os.path.isdir(source_ft) and os.path.isdir(source_ht):
        destination_ft = os.path.join(new_ft_train_folder, subfolder_name_ft)
        destination_ht = os.path.join(new_ht_train_folder, subfolder_name_ht)
        time.sleep(60)
        shutil.copytree(source_ft, destination_ft)
        shutil.copytree(source_ht, destination_ht)

# Copy and paste subfolders for the testing set
for index in test_subfolders:
    subfolder_name_ft = ft_subfolders[index]
    subfolder_name_ht = ht_subfolders[index]

    source_ft = os.path.join(original_ft_folder, subfolder_name_ft)
    source_ht = os.path.join(original_ht_folder, subfolder_name_ht)

    if os.path.isdir(source_ft) and os.path.isdir(source_ht):
        destination_ft = os.path.join(new_ft_test_folder, subfolder_name_ft)
        destination_ht = os.path.join(new_ht_test_folder, subfolder_name_ht)

        shutil.copytree(source_ft, destination_ft)
        shutil.copytree(source_ht, destination_ht)

# Copy and paste subfolders for the validation set
for index in val_subfolders:
    subfolder_name_ft = ft_subfolders[index]
    subfolder_name_ht = ht_subfolders[index]

    source_ft = os.path.join(original_ft_folder, subfolder_name_ft)
    source_ht = os.path.join(original_ht_folder, subfolder_name_ht)

    if os.path.isdir(source_ft) and os.path.isdir(source_ht):
        destination_ft = os.path.join(new_ft_val_folder, subfolder_name_ft)
        destination_ht = os.path.join(new_ht_val_folder, subfolder_name_ht)

        shutil.copytree(source_ft, destination_ft)
        shutil.copytree(source_ht, destination_ht)

# Print the randomly selected subfolders for each set
print("Training Set - Randomly selected subfolders:")
for index in selected_subfolders[:num_train_subfolders]:
    print("Subfolder", index + 1, ": FT -", ft_subfolders[index], ", HT -", ht_subfolders[index])

print("\nTesting Set - Randomly selected subfolders:")
for index in selected_subfolders[num_train_subfolders:num_train_subfolders + num_test_subfolders]:
    print("Subfolder", index + 1, ": FT -", ft_subfolders[index], ", HT -", ht_subfolders[index])

print("\nValidation Set - Randomly selected subfolders:")
for index in selected_subfolders[num_train_subfolders + num_test_subfolders:]:
    print("Subfolder", index + 1, ": FT -", ft_subfolders[index], ", HT -", ht_subfolders[index])
