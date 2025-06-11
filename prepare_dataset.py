from pathlib import Path
from PIL import Image
import re, tqdm

input_dirs = [Path("/work3/s234843/02466-Project/dataset/fluorescent/without_masks"),
             Path("/work3/s234843/02466-Project/dataset/broadband/without_masks")]
output_dirs = [Path("/work3/s234843/02466-Project/dataset/without_masks_256/fluorescent"),
              Path("/work3/s234843/02466-Project/dataset/without_masks_256/broadband")]

for i in range(len(input_dirs)):
    output_dirs[i].mkdir(parents=True, exist_ok=True)

    for tif_path in tqdm.tqdm(input_dirs[i].glob("*.tif"), desc=f"{str(output_dirs[i]).split("/")[-1]} images"):
        # Extract the time component from filename using regex
        match = re.search(r"_(\d{2})h(\d{2})m(\d{2})s_", tif_path.name)
        if not match:
            continue  # Skip files without a valid time format

        hours = int(match.group(1))
        
        if hours >= 28:
            img = Image.open(tif_path)
            img = img.resize((256, 256), Image.BICUBIC)
            out_path = output_dirs[i] / tif_path.with_suffix(".tif").name
            img.save(out_path)

    print(f"Resized images saved to: {output_dirs[i]}")

print("Finished")