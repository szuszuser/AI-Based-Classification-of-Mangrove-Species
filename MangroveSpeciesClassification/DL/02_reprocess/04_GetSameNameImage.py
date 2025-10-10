import os
from shutil import copyfile


def find_corresponding_image(label_folder, image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_files = os.listdir(label_folder)

    for label_file in label_files:
        if label_file.endswith(".tif"):  # Assuming label files are in PNG format
            image_filename = label_file.replace(".tif", ".tif")  # Get corresponding image filename
            image_path = os.path.join(image_folder, image_filename)

            if os.path.exists(image_path):
                output_image_path = os.path.join(output_folder, image_filename)
                copyfile(image_path, output_image_path)
            else:
                print(f"No corresponding image found for label {label_file}")


# Usage example
label_folder = r""  # Replace with your label files folder path
image_folder = r""  # Replace with your image files folder path
output_folder = r""  # Replace with your output folder path

find_corresponding_image(label_folder, image_folder, output_folder)
