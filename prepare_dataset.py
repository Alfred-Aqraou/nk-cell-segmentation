from pathlib import Path
from PIL import Image
import re

#input_dir = Path("/work3/s234843/02466-Project/dataset/fluorescent/without_masks")
# Input folders
input_dirs = [
    Path("/work3/s234843/02466-Project/dataset/fluorescent/without_masks"),
    Path("/work3/s234843/02466-Project/dataset/broadband/without_masks"),
    #Path("/Users/AlfredAqraouXPS15/Desktop/DTU/Year 2/Semester 4/02466 Project work - Bsc of AI and data/02466-Project/dataset/broadband/without_masks"),
    #Path("/Users/AlfredAqraouXPS15/Desktop/DTU/Year 2/Semester 4/02466 Project work - Bsc of AI and data/02466-Project/dataset/fluorescent/without_masks"),
]
output_dir = Path("/work3/s234843/02466-Project/dataset/without_masks_256_combined/dummy")
#output_dir = Path("/Users/AlfredAqraouXPS15/Desktop/DTU/Year 2/Semester 4/02466 Project work - Bsc of AI and data/02466-Project/dataset/without_masks_256_combined/dummy")
output_dir.mkdir(parents=True, exist_ok=True)

for input_dir in input_dirs:
    print(f"Processing folder: {input_dir}")
    for tif_path in input_dir.glob("*.tif"):
        # Extract the time component from filename using regex
        match = re.search(r"_(\d{2})h(\d{2})m(\d{2})s_", tif_path.name)
        if not match:
            continue  # Skip files without a valid time format

        hours = int(match.group(1))
        
        if hours >= 28:
            img = Image.open(tif_path)
            img = img.resize((256, 256), Image.BICUBIC)
            out_path = output_dir / tif_path.with_suffix(".tif").name
            img.save(out_path)

print(f"Resized images saved to: {output_dir}")