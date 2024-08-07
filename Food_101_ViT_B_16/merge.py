import os
import random
from pathlib import Path
import shutil
data_dir = Path("data/food-101/images")
merged_dir = Path("data/food-101/images/MI")
num_images_to_keep = 0
merged_dir.mkdir(parents=True, exist_ok=True)
for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        image_files = list(class_dir.glob("*.jpg"))
        random.shuffle(image_files)
        images_to_keep = image_files[:num_images_to_keep]
        for image_file in image_files[num_images_to_keep:]:
            try:
                shutil.move(image_file, merged_dir / image_file.name)
                print(f"Moved {image_file} to {merged_dir / image_file.name}")
            except Exception as e:
                print(f"Error moving {image_file}: {e}")

print("Done moving images.")