import os
import numpy as np
from PIL import Image, ImageEnhance
import pydicom


def convert_dcm_to_png(dcm_dir, size=(800, 800), dpi=300):
    """
    Convert all DCM files in the specified directory to PNG files, and save them to the folder named after the file name

    :param dcm_dir: Directory containing DCM files
    :param size: Image size, the default is (800, 800)
    :param dpi: Resolution, the default is 300
    """
    for root, dirs, files in os.walk(dcm_dir):
        for file in files:
            if file.endswith('.dcm'):
                # Read DCM file
                dcm = pydicom.dcmread(os.path.join(root, file))

                # Obtain pixel array
                pixel_arrays = dcm.pixel_array

                # Obtain window width and window level
                window_width = dcm.WindowWidth
                window_center = dcm.WindowCenter

                # Create a new directory
                new_dir = os.path.join(root, os.path.splitext(file)[0])
                os.makedirs(new_dir, exist_ok=True)

                # Convert pixel array and save as PNG file
                for i, pixel_array in enumerate(pixel_arrays):
                    # Convert pixel array to grayscale image
                    pixel_array = np.array(pixel_array)
                    pixel_array = ((pixel_array - window_center + 0.5 * window_width) / window_width) * 255
                    pixel_array = np.clip(pixel_array, 0, 255).astype('uint8')
                    # Find the brightest pixel value in a pixel array
                    max_pixel_value = np.max(pixel_array)
                    # Increase the pixel value of all white parts in the pixel array
                    threshold = 1
                    brightened_array = np.where(pixel_array > threshold, pixel_array + (255 - max_pixel_value), pixel_array)

                    image = Image.fromarray(pixel_array, mode='L')

                    # Change picture size and resolution
                    image = image.resize(size, resample=Image.LANCZOS)
                    image.info['dpi'] = dpi, dpi
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(1.2)

                    # Generate filename and save PNG file
                    filename = f'{os.path.splitext(file)[0]}_{i + 1}.png'
                    filepath = os.path.join(new_dir, filename)
                    image.save(filepath)

                    print(f'Saved {filename}')


if __name__ == '__main__':
    # Set the directory where the DCM file needs to be converted
    dcm_dir = '/path/'
    convert_dcm_to_png(dcm_dir)

