"""
clustering_analysis.py

K-Means clustering on ResNet50 features, with a proper quantitative
evaluation against ground-truth labels -- the piece that was missing
from the original KMeans.ipynb.

Fixes relative to the original notebook:
1. VARIABLE COLLISION: the original notebook reused the name `labels`
   for both the true class labels (from the file loader) and the
   K-Means cluster assignments, silently destroying the ground truth
   labels needed for evaluation. This version keeps them as separate,
   clearly-named arrays throughout.
2. QUANTITATIVE EVALUATION: the original notebook only inspected
   cluster membership visually (sample images per cluster). This adds
   Adjusted Rand Index, cluster purity, and a cluster-vs-true-label
   contingency table/matrix -- an actual measurement of whether the
   "no overlap" observation holds, not just a manual visual check.
3. MEMORY: features were already extracted batch-wise by
   extract_features.py (this script only loads the resulting small
   feature arrays, not the raw images).

Run from inside the Brain_Cancer/ directory, after extract_features.py
has completed.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "outputs" / "clustering"
OUTPUT_DIR = FEATURES_DIR
K = 3
CLASS_DISPLAY_NAMES = {"brain_glioma": "Glioma", "brain_menin": "Menin", "brain_tumor": "Tumor"}


def group_by_image_id(features, file_names, image_ids, true_labels):
    """Average features for all crops sharing the same image_id, and
    carry along a single true label per grouped image (all crops of
    the same image_id share a class by construction)."""
    unique_ids = sorted(set(image_ids))
    grouped_features = np.zeros((len(unique_ids), features.shape[1]), dtype="float32")
    grouped_true_labels = []
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}

    counts = np.zeros(len(unique_ids))
    label_for_id = {}
    for i, uid in enumerate(image_ids):
        gi = id_to_idx[uid]
        grouped_features[gi] += features[i]
        counts[gi] += 1
        label_for_id[uid] = true_labels[i]  # all crops share the same class

    grouped_features /= counts[:, None]
    grouped_true_labels = np.array([label_for_id[uid] for uid in unique_ids])

    return grouped_features, grouped_true_labels, np.array(unique_ids)


def main():
    features = np.load(FEATURES_DIR / "features.npy")
    file_names = np.load(FEATURES_DIR / "file_names.npy")
    image_ids = np.load(FEATURES_DIR / "image_ids.npy")
    true_labels_raw = np.load(FEATURES_DIR / "true_labels.npy")

    print(f"Loaded {features.shape[0]} raw feature vectors ({features.shape[1]}-dim).")

    grouped_features, grouped_true_labels, unique_ids = group_by_image_id(
        features, file_names, image_ids, true_labels_raw
    )
    print(f"Grouped into {grouped_features.shape[0]} unique scans.")
    for cls in sorted(set(grouped_true_labels)):
        print(f"  {cls}: {(grouped_true_labels == cls).sum()}")

    # ---- Elbow method ----
    wcss = []
    k_range = range(1, 15)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(grouped_features)
        wcss.append(km.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(list(k_range), wcss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_plot.png", dpi=150)
    plt.close()

    # ---- Final K-Means fit ----
    kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    cluster_assignments = kmeans.fit_predict(grouped_features)  # kept separate from true labels

    cluster_sizes = {int(c): int((cluster_assignments == c).sum()) for c in range(K)}
    print(f"\nCluster sizes: {cluster_sizes}")

    # ---- Quantitative evaluation against ground truth ----
    ari = adjusted_rand_score(grouped_true_labels, cluster_assignments)
    print(f"Adjusted Rand Index: {ari:.4f}")

    # Contingency table: rows = true class, cols = cluster
    class_names_sorted = sorted(set(grouped_true_labels))
    cm_raw = contingency_matrix(grouped_true_labels, cluster_assignments)

    # Hungarian algorithm: find the best 1-1 mapping of clusters -> classes
    # to make the contingency table readable as an aligned confusion matrix,
    # and to compute a purity-style "accuracy" if clusters are treated as
    # predicted classes.
    row_ind, col_ind = linear_sum_assignment(-cm_raw)  # maximize matches
    cluster_to_class = {int(col): str(class_names_sorted[row]) for row, col in zip(row_ind, col_ind)}
    print(f"Best cluster -> class alignment: {cluster_to_class}")

    aligned_preds = np.array([cluster_to_class.get(c, "unmatched") for c in cluster_assignments])
    purity_correct = (aligned_preds == grouped_true_labels).sum()
    purity = purity_correct / len(grouped_true_labels)
    print(f"Cluster purity (best-alignment accuracy): {purity:.4f} "
          f"({purity_correct}/{len(grouped_true_labels)})")

    # Confusion matrix using the Hungarian-aligned labels, in display order
    display_order = [c for c in class_names_sorted]
    cm_aligned = confusion_matrix(grouped_true_labels, aligned_preds, labels=display_order)
    disp_labels = [CLASS_DISPLAY_NAMES.get(c, c) for c in display_order]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_aligned, display_labels=disp_labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"K-Means Clusters vs. True Labels (best alignment)\nARI = {ari:.3f}, Purity = {purity:.3f}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_vs_true_label_confusion_matrix.png", dpi=150)
    plt.close()

    # ---- Save sample images per cluster (for visual inspection) ----
    # file_names is (n_raw_crops, 2): [class, filename]. Map grouped image_id
    # back to one representative raw file for display.
    id_to_one_file = {}
    for i, uid in enumerate(image_ids):
        if uid not in id_to_one_file:
            id_to_one_file[uid] = (file_names[i][0], file_names[i][1])  # (class, filename)

    for cluster_id in range(K):
        members = unique_ids[cluster_assignments == cluster_id][:5]
        fig, axes = plt.subplots(1, len(members), figsize=(3 * len(members), 3.2))
        if len(members) == 1:
            axes = [axes]
        for ax, uid in zip(axes, members):
            cls, fname = id_to_one_file[uid]
            img = plt.imread(BASE_DIR / cls / fname)
            ax.imshow(img)
            ax.set_title(CLASS_DISPLAY_NAMES.get(cls, cls), fontsize=10)
            ax.axis("off")
        plt.suptitle(f"Cluster {cluster_id} — {cluster_sizes[cluster_id]} scans "
                     f"(aligned to: {CLASS_DISPLAY_NAMES.get(cluster_to_class.get(cluster_id, '?'), '?')})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cluster_{cluster_id}_samples.png", dpi=150)
        plt.close()

    # ---- Save metrics ----
    metrics = {
        "n_grouped_scans": int(grouped_features.shape[0]),
        "k": K,
        "cluster_sizes": cluster_sizes,
        "adjusted_rand_index": float(ari),
        "cluster_purity": float(purity),
        "cluster_to_class_alignment": cluster_to_class,
        "contingency_table": {
            "true_label_order": class_names_sorted,
            "cluster_order": list(range(K)),
            "table": cm_raw.tolist(),
        },
        "elbow_wcss": {str(k): float(w) for k, w in zip(k_range, wcss)},
    }
    with open(OUTPUT_DIR / "clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    main()
