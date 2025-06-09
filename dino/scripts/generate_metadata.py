import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

data_dir = "/work3/s234843/02466-Project/dataset"
root_dir = os.path.join(data_dir, "fluorescent/without_masks_224")
extra_dir = os.path.join(data_dir, "extra")
os.makedirs(extra_dir, exist_ok=True)

# Collect all image paths
image_paths = sorted(glob(os.path.join(root_dir, "*.tif")))

# Split into TRAIN and VAL (e.g., 80/20)
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

def write_split(split_name, paths):
    entries = [(p, 0) for p in paths]
    class_ids = [0] * len(paths)
    class_names = ["fluorescent"]
    
    np.save(os.path.join(extra_dir, f"entries-{split_name}.npy"), np.array(entries, dtype=object))
    np.save(os.path.join(extra_dir, f"class-ids-{split_name}.npy"), np.array(class_ids))
    np.save(os.path.join(extra_dir, f"class-names-{split_name}.npy"), np.array(class_names))

# Save both TRAIN and VAL
write_split("TRAIN", train_paths)
write_split("VAL", val_paths)

print(f"TRAIN and VAL metadata saved to: {extra_dir}")
