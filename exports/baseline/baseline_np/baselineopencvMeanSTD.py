import numpy as np
import os

# ----------- Settings -----------
folder = "exports/baseline/baseline_np"
filenames = [f"per_image_iou_fold{i}_baseline.npy" for i in range(1, 6)]

# ----------- Load and Aggregate -----------
all_ious = []

for filename in filenames:
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"❌ File missing: {filename}")
        continue

    ious = np.load(path)$
    print(f"Loaded {len(ious)} IoUs from {filename}")
    all_ious.extend(ious)

# ----------- Compute Stats -----------
if len(all_ious) == 0:
    print("No data loaded.")
else:
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious, ddof=1)

    print("\n=== Baseline Performance Summary ===")
    print(f"Total IoUs loaded: {len(all_ious)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Standard Deviation: {std_iou:.4f}")