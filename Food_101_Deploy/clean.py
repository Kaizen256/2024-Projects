import os
import random
from pathlib import Path
import shutil

# Set the root directory of the Food101 dataset
data_dir = Path("data/food-101/images")

# Number of images to keep per class
num_images_to_keep = 1

# Iterate through each class directory
for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        # Get all image file paths in the class directory
        image_files = list(class_dir.glob("*.jpg"))
        
        # Shuffle the list of image files
        random.shuffle(image_files)
        
        # Keep only the specified number of images
        images_to_keep = image_files[:num_images_to_keep]
        
        # Delete the rest of the images
        for image_file in image_files[num_images_to_keep:]:
            try:
                image_file.unlink()  # Delete the image file
                print(f"Deleted {image_file}")
            except Exception as e:
                print(f"Error deleting {image_file}: {e}")

print("Done cleaning up dataset.")