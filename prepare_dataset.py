from pathlib import Path
from PIL import Image
import re

input_dir = Path("dataset/fluorescent/without_masks")
output_dir = Path("dataset/fluorescent/without_masks_256")
output_dir.mkdir(parents=True, exist_ok=True)

for tif_path in input_dir.glob("*.tif"):
    # Extract the time component from filename using regex
    match = re.search(r"_(\d{2})h(\d{2})m(\d{2})s_", tif_path.name)
    if not match:
        continue  # Skip files without a valid time format

    hours = int(match.group(1))
    
    if hours >= 29:
        img = Image.open(tif_path)
        img = img.resize((256, 256), Image.BICUBIC)
        out_path = output_dir / tif_path.with_suffix(".tif").name
        img.save(out_path)

print(f"Resized images saved to: {output_dir}")
