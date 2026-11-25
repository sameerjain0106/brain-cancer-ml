"""Evaluate PCA compression on the same saved ResNet50 novelty task.

From the project root, run: python web/tools/benchmark-novelty.py
This retrospective benchmark uses saved features and does not read any scans.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "outputs" / "clustering"
OUTPUT = ROOT / "web" / "public" / "research" / "novelty-benchmark.json"
BASELINE_LABEL = "brain_glioma"
SEED = 42
PCA_DIMENSIONS = 64
CALIBRATION_QUANTILE = 0.90


def load_grouped_features():
    features = np.load(FEATURES_DIR / "features.npy", mmap_mode="r")
    identifiers = np.load(FEATURES_DIR / "image_ids.npy")
    labels = np.load(FEATURES_DIR / "true_labels.npy")
    if features.ndim != 2 or len(features) != len(identifiers) or len(features) != len(labels):
        raise ValueError("The saved ResNet50 embeddings and labels are not aligned.")

    group_keys = np.stack((labels, identifiers), axis=1)
    groups, positions, sizes = np.unique(
        group_keys, axis=0, return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(groups), features.shape[1]), dtype=np.float64)
    np.add.at(sums, positions, features)
    grouped = sums / sizes[:, None]
    if not np.isfinite(grouped).all():
        raise ValueError("The saved ResNet50 embeddings contain non-finite values.")
    return grouped, groups[:, 0], len(features)


def unit_rows(features):
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError("A zero-norm group cannot be compared by cosine distance.")
    return features / lengths


def evaluate(train, calibration, test, actual):
    train_vectors = unit_rows(train)

    def distances(queries):
        similarities = unit_rows(queries) @ train_vectors.T
        return 1 - np.clip(similarities.max(axis=1), -1, 1)

    threshold = np.quantile(distances(calibration), CALIBRATION_QUANTILE)
    scores = distances(test)
    predicted = scores > threshold
    counts = confusion_matrix(actual, predicted, labels=[False, True])
    tn, fp, fn, tp = counts.ravel()
    return {
        "dimension": train.shape[1],
        "threshold": round(float(threshold), 8),
        "auroc": round(float(roc_auc_score(actual, scores)), 6),
        "balancedAccuracy": round(float(balanced_accuracy_score(actual, predicted)), 6),
        "sensitivity": round(float(tp / (tp + fn)), 6),
        "specificity": round(float(tn / (tn + fp)), 6),
        "confusionMatrix": counts.tolist(),
    }


def main():
    features, labels, image_count = load_grouped_features()
    inlier_groups = np.flatnonzero(labels == BASELINE_LABEL)
    other_groups = np.flatnonzero(labels != BASELINE_LABEL)
    shuffled = np.random.default_rng(SEED).permutation(inlier_groups)
    train_end = round(0.60 * len(shuffled))
    calibration_end = round(0.80 * len(shuffled))

    train = features[shuffled[:train_end]]
    calibration = features[shuffled[train_end:calibration_end]]
    test_indices = np.concatenate((shuffled[calibration_end:], other_groups))
    test = features[test_indices]
    actual = labels[test_indices] != BASELINE_LABEL

    full = evaluate(train, calibration, test, actual)
    compressor = PCA(
        n_components=PCA_DIMENSIONS, svd_solver="randomized", random_state=SEED
    ).fit(train)
    mean = compressor.mean_
    centered = evaluate(train - mean, calibration - mean, test - mean, actual)
    compressed = evaluate(
        compressor.transform(train),
        compressor.transform(calibration),
        compressor.transform(test),
        actual,
    )
    compressed["explainedVarianceRatio"] = round(
        float(compressor.explained_variance_ratio_.sum()), 6
    )

    result = {
        "version": 1,
        "title": "ResNet50 novelty scoring with and without PCA",
        "scope": "Exploratory retrospective benchmark on the project dataset",
        "unit": "Filename-derived scan group, not a verified patient ID",
        "baselineLabel": BASELINE_LABEL,
        "imageCount": image_count,
        "groupCount": len(features),
        "split": {
            "seed": SEED,
            "trainGliomaGroups": len(train),
            "calibrationGliomaGroups": len(calibration),
            "testGliomaGroups": int((~actual).sum()),
            "testOtherGroups": len(other_groups),
            "testOtherGroupsByClass": {
                label: int((labels == label).sum())
                for label in np.unique(labels)
                if label != BASELINE_LABEL
            },
        },
        "method": {
            "features": "Mean saved ImageNet ResNet50 embedding across crops in each scan group",
            "score": "One minus cosine similarity to the nearest reference training group",
            "threshold": "90th percentile of held-out glioma calibration scores",
            "centering": "Subtract the glioma training mean and keep all dimensions",
            "pca": f"{PCA_DIMENSIONS} centered components fit on glioma training groups only",
            "confusionMatrixOrder": ["Glioma reference group", "Other dataset group"],
            "note": (
                "All branches use the same train, calibration and test groups and scoring rule. "
                "Centering alone alters the cosine geometry, so differences between full and PCA "
                "features cannot be attributed to reducing dimensions alone. The original grayscale "
                "autoencoder experiment used a different representation, split and scorer. "
                "This benchmark does not establish performance on new scanners."
            ),
        },
        "results": {
            "full": full,
            "centered": centered,
            "pca64": compressed,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
