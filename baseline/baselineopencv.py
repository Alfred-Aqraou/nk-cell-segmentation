import os
import numpy as np
import cv2
import csv
from cell_tools import find_NKcell_cluster_centers

# ----------- Paths -------------
labels_folder = "/Users/aliafif/Desktop/Code/Fagprojekt/02466-Project/dataset/fluorescent/with_masks/masks"
image_folder = "/Users/aliafif/Desktop/Code/Fagprojekt/02466-Project/dataset/fluorescent/with_masks/images"
export_dir = "exports/baseline/"
os.makedirs(export_dir, exist_ok=True)

# ----------- Load labeled image filenames -------------
label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('GT.tif')])
print(f"🔍 Found {len(label_files)} labeled images.")

# ----------- Evaluate baseline method -------------
iou_list = []
skipped_files = []

for filename in label_files:
    label_path = os.path.join(labels_folder, filename)

    if not os.path.exists(label_path):
        skipped_files.append(filename)
        continue

    labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if labels is None:
        skipped_files.append(filename)
        continue

    gt_binary = (labels > 0).astype(np.uint8)
    raw_filename = filename.replace('_GT', '')
    image_path = os.path.join(image_folder, raw_filename)

    if not os.path.exists(image_path):
        skipped_files.append(raw_filename)
        continue

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        skipped_files.append(raw_filename)
        continue

    try:
        _, _, cell_coordinates, _, _, _, _ = find_NKcell_cluster_centers(image, plot_steps=False)
        pred_binary = (cell_coordinates > 0).astype(np.uint8)

        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = gt_binary.sum() + pred_binary.sum() - intersection
        iou = intersection / (union + 1e-8)

        img_id = os.path.basename(image_path)
        iou_list.append((img_id, iou))
    except Exception as e:
        print(f"❗ Error processing {filename}: {e}")
        skipped_files.append(filename)

print(f"\nProcessed {len(iou_list)} images, Skipped {len(skipped_files)} files.")
if skipped_files:
    print(f"⚠️ Skipped files:\n{skipped_files}")

# ----------- Compute and save results -------------
ious = [iou for _, iou in iou_list]
mean_iou = np.mean(ious)
std_iou = np.std(ious)

# Save per-image IoUs
npy_path = os.path.join(export_dir, "per_image_iou_baseline.npy")
np.save(npy_path, np.array(iou_list, dtype=object))

# Save summary CSV
csv_path = os.path.join(export_dir, "results_baseline.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["mean_iou", "std_iou"])
    writer.writeheader()
    writer.writerow({"mean_iou": mean_iou, "std_iou": std_iou})

# ----------- Print Summary -------------
print(f"\n Overall Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
print(f" Saved per-image results to {npy_path}")
print(f"Saved summary to {csv_path}")