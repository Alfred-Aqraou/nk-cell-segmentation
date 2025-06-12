import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
from cell_tools import find_NKcell_cluster_centers

# ----------- Paths -------------

labels_folder = "/Users/aliafif/Desktop/Code/Fagprojekt/02466-Project/dataset/fluorescent/with_masks/masks"
image_folder  = "/Users/aliafif/Desktop/Code/Fagprojekt/02466-Project/dataset/fluorescent/with_masks/images"

# ----------- Load labeled images -------------

label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('GT.tif')])
labeled_indices = list(range(len(label_files)))

print(f"Found {len(label_files)} labeled images.")

# ----------- Set up KFold -------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ----------- Main loop -------------

for fold, (train_idx, val_idx) in enumerate(kf.split(labeled_indices)):
    print(f"\n=== Fold {fold+1} ===")
    fold_iou_list = []
    skipped_files = []

    for idx in val_idx:
        filename = label_files[idx]
        label_path = os.path.join(labels_folder, filename)

        if not os.path.exists(label_path):
            print(f"❌ GT file missing: {label_path}")
            skipped_files.append(filename)
            continue

        labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if labels is None:
            print(f"⚠️ Could not read GT image: {label_path}")
            skipped_files.append(filename)
            continue

        gt_binary = (labels > 0).astype(np.uint8)

        raw_filename = filename.replace('_GT', '')
        image_path = os.path.join(image_folder, raw_filename)

        if not os.path.exists(image_path):
            print(f"❌ Raw image missing: {image_path}")
            skipped_files.append(raw_filename)
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠️ Could not read raw image: {image_path}")
            skipped_files.append(raw_filename)
            continue

        print(f"🟢 Processing: {filename} + {raw_filename}")

        try:
            _, _, cell_coordinates, _, _, _, _ = find_NKcell_cluster_centers(image, plot_steps=False)
            pred_binary = (cell_coordinates > 0).astype(np.uint8)

            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = gt_binary.sum() + pred_binary.sum() - intersection
            iou = intersection / (union + 1e-8)

            fold_iou_list.append(iou)
        except Exception as e:
            print(f"❗ Error processing {filename}: {e}")
            skipped_files.append(filename)

    print(f"✅ Fold {fold+1}: {len(fold_iou_list)} processed, {len(skipped_files)} skipped.")
    if skipped_files:
        print(f"⚠️ Skipped files:\n{skipped_files}")

    export_dir = "exports/baseline/baseline_np"
    os.makedirs(export_dir, exist_ok=True)
    save_path = os.path.join(export_dir, f"per_image_iou_fold{fold+1}_baseline.npy")
    np.save(save_path, np.array(fold_iou_list))
    print(f"💾 Saved fold {fold+1} IoUs to {save_path}")
