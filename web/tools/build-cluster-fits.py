"""Export the archived MRI K-means fits in the browser map's group order."""

import json
from math import isclose
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
CLUSTERING = ROOT / "outputs" / "clustering"
PUBLIC = ROOT / "web" / "public" / "research"


def main():
    reference = json.loads((PUBLIC / "reference.json").read_text(encoding="utf-8"))
    archive = json.loads((CLUSTERING / "clustering_metrics.json").read_text(encoding="utf-8"))
    features = np.load(CLUSTERING / "features.npy", allow_pickle=False)
    image_ids = np.load(CLUSTERING / "image_ids.npy", allow_pickle=False)
    labels = np.load(CLUSTERING / "true_labels.npy", allow_pickle=False)
    if features.shape != (len(image_ids), reference["dimension"]) or labels.shape != image_ids.shape:
        raise ValueError("Archived crop features, scan IDs, and labels do not align")

    # The notebook sums raw float32 crop vectors in archive order, then fits
    # sorted scan means. The browser's unit vectors are for cosine matching.
    ids, row_to_group, counts = np.unique(image_ids, return_inverse=True, return_counts=True)
    grouped = np.zeros((len(ids), features.shape[1]), dtype=np.float32)
    np.add.at(grouped, row_to_group, features)
    grouped /= counts[:, None].astype(np.float64)

    group_ids = [group["id"] for group in reference["groups"]]
    group_by_id = {group["id"]: group for group in reference["groups"]}
    if len(group_ids) != len(ids) or set(group_ids) != set(ids) or len(ids) != archive["n_grouped_scans"]:
        raise ValueError("The archive and browser map do not contain the same scan groups")
    browser_order = np.searchsorted(ids, group_ids)
    grouped_labels = np.array([group_by_id[scan_id]["label"] for scan_id in ids])
    if not np.array_equal(grouped_labels[row_to_group], labels):
        raise ValueError("Archived crop labels disagree with browser scan-group labels")
    if any(group["sourceCount"] != counts[index] for group, index in zip(reference["groups"], browser_order)):
        raise ValueError("Archived crop counts disagree with browser scan-group counts")

    fits = {}
    with threadpool_limits(limits=1):
        for k in range(1, 15):
            model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(grouped)
            expected_wcss = archive["elbow_wcss"][str(k)]
            # Allow small float32 reduction differences across BLAS builds.
            if not isclose(model.inertia_, expected_wcss, rel_tol=1e-6):
                raise ValueError(f"k={k} inertia {model.inertia_} differs from archived {expected_wcss}")
            if len(np.unique(model.labels_)) != k:
                raise ValueError(f"k={k} produced fewer than {k} populated clusters")
            if k == archive["k"]:
                ari = adjusted_rand_score(grouped_labels, model.labels_)
                if not isclose(ari, archive["adjusted_rand_index"], abs_tol=1e-12):
                    raise ValueError(f"k={k} adjusted Rand index {ari} differs from the archive")
                aligned_counts = {
                    frozenset(grouped_labels[model.labels_ == cluster]): int((model.labels_ == cluster).sum())
                    for cluster in range(k)
                }
                archived_counts = {
                    frozenset({label}): archive["cluster_sizes"][cluster]
                    for cluster, label in archive["cluster_to_class_alignment"].items()
                }
                if aligned_counts != archived_counts:
                    raise ValueError(f"k={k} cluster alignment differs from the archived partition")
            fits[str(k)] = {
                "assignments": model.labels_[browser_order].tolist(),
                "wcss": model.inertia_,
            }

    result = {"version": 1, "groupIds": group_ids, "fits": fits}
    (PUBLIC / "cluster-fits.json").write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validated and exported {len(fits)} K-means fits for {len(group_ids)} scan groups.")


if __name__ == "__main__":
    main()
