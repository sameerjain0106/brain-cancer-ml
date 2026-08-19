"""
extract_features.py

Extracts ResNet50 (ImageNet, avg-pooled) features for every image in the
Brain_Cancer dataset. This replaces the original notebook's approach of
loading all images into one big float32 array before running the model --
that pattern needs ~5GB+ of RAM for this dataset and does not scale to
larger datasets. This version processes images in small batches, holding
at most one batch of raw pixels in memory at a time.

Methodology is otherwise unchanged from the original notebook:
- ResNet50, ImageNet weights, include_top=False, pooling='avg' -> 2048-dim feature per image
- Images loaded at their native 512x512 resolution (matches original)
- preprocess_input applied identically to the original

This version is also CHECKPOINTED / RESUMABLE: on a normal machine this
runs start-to-finish in one call, but on constrained/CPU-only hardware a
single run of ~6,000 ResNet50 forward passes can take significant wall
time, so this script saves progress and can be safely re-run to pick up
where it left off (controlled by MAX_SECONDS_PER_RUN below).

Run from inside the Brain_Cancer/ directory (where this script and the
brain_tumor/, brain_menin/, brain_glioma/ folders live side by side).
"""
import os
import re
import time
import json
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CLASSES = ["brain_tumor", "brain_menin", "brain_glioma"]
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
MAX_SECONDS_PER_RUN = 150  # time budget per invocation; re-run the script to resume
CHECKPOINT_EVERY_SECONDS = 30  # save progress this often within a run, not just at the end
OUTPUT_DIR = BASE_DIR / "outputs" / "clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = OUTPUT_DIR / "features.npy"
META_PATH = OUTPUT_DIR / "extraction_meta.json"


def list_image_files():
    """Collect (class, filename) pairs for every image, and derive a
    unique per-scan image_id the same way the original notebook did
    (grouping multiple crops of the same underlying scan)."""
    records = []
    for cls in CLASSES:
        folder = BASE_DIR / cls
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                match = re.findall(r"(\d{3})", fname)
                image_id = f"{cls}_{match[0]}" if match else f"{cls}_{fname}"
                records.append({"cls": cls, "fname": fname, "image_id": image_id})
    return records


def main():
    records = list_image_files()
    n_total = len(records)

    # ---- Resume from checkpoint if one exists ----
    if FEATURES_PATH.exists() and META_PATH.exists():
        all_features = np.load(FEATURES_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        start_idx = meta["n_processed"]
        print(f"Resuming from checkpoint: {start_idx}/{n_total} already done.")
    else:
        all_features = np.zeros((n_total, 2048), dtype="float32")
        start_idx = 0
        print(f"Starting fresh extraction. Found {n_total} images across {len(CLASSES)} classes.")

    if start_idx >= n_total:
        print("Already complete.")
        _finalize(records)
        return

    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    t_start = time.time()
    t_last_checkpoint = t_start
    idx = start_idx
    while idx < n_total:
        if time.time() - t_start > MAX_SECONDS_PER_RUN:
            break

        batch_records = records[idx : idx + BATCH_SIZE]
        imgs = []
        for r in batch_records:
            img_path = BASE_DIR / r["cls"] / r["fname"]
            img = kimage.load_img(img_path, target_size=IMG_SIZE)
            imgs.append(kimage.img_to_array(img))
        batch_arr = np.array(imgs, dtype="float32")
        batch_arr = preprocess_input(batch_arr)

        feats = model.predict(batch_arr, batch_size=BATCH_SIZE, verbose=0)
        all_features[idx : idx + len(batch_records)] = feats
        idx += len(batch_records)

        if time.time() - t_last_checkpoint > CHECKPOINT_EVERY_SECONDS:
            np.save(FEATURES_PATH, all_features)
            with open(META_PATH, "w") as f:
                json.dump({"n_processed": idx, "n_total": n_total}, f)
            t_last_checkpoint = time.time()
            print(f"  checkpoint: {idx}/{n_total}", flush=True)

    elapsed = time.time() - t_start
    rate = (idx - start_idx) / elapsed if elapsed > 0 else 0
    remaining_min = (n_total - idx) / rate / 60 if rate > 0 else float("inf")
    print(f"Processed {idx - start_idx} images this run in {elapsed/60:.1f} min "
          f"({idx}/{n_total} total done, ~{remaining_min:.1f} min remaining)")

    np.save(FEATURES_PATH, all_features)
    with open(META_PATH, "w") as f:
        json.dump({"n_processed": idx, "n_total": n_total}, f)

    if idx >= n_total:
        print("\nExtraction complete.")
        _finalize(records)
    else:
        print("Run this script again to continue from checkpoint.")


def _finalize(records):
    file_names = np.array([(r["cls"], r["fname"]) for r in records])
    image_ids = np.array([r["image_id"] for r in records])
    true_labels = np.array([r["cls"] for r in records])
    np.save(OUTPUT_DIR / "file_names.npy", file_names)
    np.save(OUTPUT_DIR / "image_ids.npy", image_ids)
    np.save(OUTPUT_DIR / "true_labels.npy", true_labels)
    print(f"Saved metadata arrays to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

