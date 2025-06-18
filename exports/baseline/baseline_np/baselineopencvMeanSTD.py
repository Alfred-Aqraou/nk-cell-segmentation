import numpy as np
import os

# ----------- Settings -----------
filenames = [f"exports/baseline/baseline_np/per_image_iou_fold{i}_baseline.npy" for i in range(1, 6)]

# ----------- Load and Aggregate -----------
all_ious = []

for filename in filenames:
    if not os.path.exists(filename):
        print(f"❌ File missing: {filename}")
        continue

    ious = np.load(filename, allow_pickle=True)
    print(f"Loaded {len(ious)} IoUs from {filename}")
    all_ious.extend([float(iou) if isinstance(iou, (float, np.float32, np.float64)) else float(iou[1]) for iou in ious])

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