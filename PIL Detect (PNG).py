# Python PIL Detect if an image is completely black or white and Detect images removal

import os
from PIL import Image

def is_image_black_white(image_path):
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    colors = image.getcolors()  # Get color histogram
    num_colors = len(colors)
    return num_colors <= 2  # If only black and white present, it's completely black or white

def remove_black_white_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):  # Check if file is a PNG image
                image_path = os.path.join(root, file)
                if is_image_black_white(image_path):
                    os.remove(image_path)  # Remove the image file

# Specify the folder path containing subfolders with PNG images
folder_path = '/path'

remove_black_white_images(folder_path)

# Provide the folder path containing the BMP images
folder_path = "/path"
remove_black_white_images(folder_path)

