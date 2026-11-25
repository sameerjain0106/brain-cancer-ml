"""Export a compact browser reference index from the archived MRI experiment."""

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
CLUSTERING = ROOT / "outputs" / "clustering"
ANOMALY = ROOT / "outputs" / "anomaly_detection"
PUBLIC = ROOT / "web" / "public" / "research"
DIMENSION = 2048
LABELS = ("brain_glioma", "brain_menin", "brain_tumor")
LABEL_TITLES = {
    "brain_glioma": "Glioma",
    "brain_menin": "Meningioma",
    "brain_tumor": "Tumor source label",
}
# The original figure generator uses the same layout for all three sheets.
# The first panel shows the first crop of the first sorted scan ID in that cluster.
FIRST_PANEL = (41, 72, 427, 457)
FIGURES = {
    "cluster_0_samples.png": CLUSTERING / "cluster_0_samples.png",
    "cluster_1_samples.png": CLUSTERING / "cluster_1_samples.png",
    "cluster_2_samples.png": CLUSTERING / "cluster_2_samples.png",
    "cluster_vs_true_label_confusion_matrix.png":
        CLUSTERING / "cluster_vs_true_label_confusion_matrix.png",
    "elbow_plot.png": CLUSTERING / "elbow_plot.png",
    "clustering_metrics.json": CLUSTERING / "clustering_metrics.json",
    "anomaly_confusion_matrix.png": ANOMALY / "confusion_matrix.png",
    "anomaly_metrics.json": ANOMALY / "metrics.json",
    "distance_distribution.png": ANOMALY / "distance_distribution.png",
    "training_loss.png": ANOMALY / "training_loss.png",
}


def unit_rows(vectors):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Reference vectors must have finite, nonzero norm")
    return (vectors / lengths).astype(np.float32)


def load_features():
    vectors = np.load(CLUSTERING / "features.npy", allow_pickle=False)
    scan_ids = np.load(CLUSTERING / "image_ids.npy", allow_pickle=False)
    filenames = np.load(CLUSTERING / "file_names.npy", allow_pickle=False)
    labels = np.load(CLUSTERING / "true_labels.npy", allow_pickle=False)
    counts = json.loads((CLUSTERING / "extraction_meta.json").read_text(encoding="utf-8"))
    if counts["n_total"] != counts["n_processed"]:
        raise ValueError("Feature extraction is incomplete")
    count = counts["n_total"]
    if not (
        vectors.shape == (count, DIMENSION)
        and scan_ids.shape == labels.shape == (count,)
        and filenames.shape == (count, 2)
        and np.array_equal(filenames[:, 0], labels)
        and set(labels.tolist()) == set(LABELS)
    ):
        raise ValueError("Archived feature, file, group, or label arrays do not align")
    derived_ids = []
    # Match extract_features.py, including its class-prefixed filename heuristic.
    for label, filename in filenames:
        match = re.search(r"(\d{3})", filename)
        part = match.group(1) if match else filename
        derived_ids.append(f"{label}_{part}")
    if not np.array_equal(derived_ids, scan_ids):
        raise ValueError("Archived scan IDs do not match the source filename rule")
    return vectors.astype(np.float32, copy=False), scan_ids, labels


def build_groups(vectors, scan_ids, labels):
    ids, row_to_group, counts = np.unique(
        scan_ids, return_inverse=True, return_counts=True
    )
    centroids = np.zeros((len(ids), DIMENSION), dtype=np.float32)
    np.add.at(centroids, row_to_group, vectors)
    centroids /= counts[:, None]
    group_labels = []
    for index in range(len(ids)):
        members = np.unique(labels[row_to_group == index])
        if len(members) != 1:
            raise ValueError(f"Conflicting source labels within scan ID {ids[index]}")
        group_labels.append(str(members[0]))
    return ids, row_to_group, counts, group_labels, unit_rows(centroids)


def validate_neighbors(crop_vectors, row_to_group, crop_labels, group_vectors, group_labels):
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    nearest_distances = []
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    group_label_indices = np.array([label_to_index[label] for label in group_labels])
    for start in range(0, len(crop_vectors), 256):
        end = min(start + 256, len(crop_vectors))
        scores = crop_vectors[start:end] @ group_vectors.T
        # A complete source group stays out for each crop. This mirrors the
        # single-crop browser query without comparing against its own crops.
        scores[np.arange(end - start), row_to_group[start:end]] = -np.inf
        winners = np.argmax(scores, axis=1)
        distances = 1 - scores[np.arange(end - start), winners]
        nearest_distances.extend(np.clip(distances, 0, 2).tolist())
        truths = [label_to_index[label] for label in crop_labels[start:end]]
        np.add.at(confusion, (truths, group_label_indices[winners]), 1)
    count, correct = int(confusion.sum()), int(np.trace(confusion))
    return {
        "unit": "scan ID",
        "groupCount": len(group_vectors),
        "count": count,
        "correct": correct,
        "accuracy": correct / count,
        "labels": list(LABELS),
        "confusionMatrix": confusion.tolist(),
        "nearestDistance": {
            **{f"p{q}": float(np.percentile(nearest_distances, q)) for q in (50, 90, 95)},
            "max": float(max(nearest_distances)),
        },
        "scope": (
            "Each archived crop was matched to the nearest cosine-distance reference "
            "after excluding all crops with its scan ID. Counts and accuracy describe "
            "source-label agreement on this collection."
        ),
        "grouping": (
            "A scan ID is inferred from the source label and filename. These IDs "
            "do not establish patient-level independence."
        ),
        "distanceExplanation": (
            "Cosine distance is 1 minus cosine similarity in the ResNet50 feature space. "
            "The reference range comes from same-collection held-out crop comparisons. "
            "A larger distance means less resemblance to this index."
        ),
    }


def build_samples(archive, group_labels, row_to_group, crop_vectors):
    alignment = archive["cluster_to_class_alignment"]
    order = archive["contingency_table"]["true_label_order"]
    contingency = np.array(archive["contingency_table"]["table"])
    cluster_order = archive["contingency_table"]["cluster_order"]
    samples = []
    crops = []
    for cluster, label in sorted(alignment.items(), key=lambda item: int(item[0])):
        cluster_column = cluster_order.index(int(cluster))
        label_row = order.index(label)
        if contingency[:, cluster_column].sum() != contingency[label_row, cluster_column]:
            raise ValueError("Cannot identify first sheet panel from pure-cluster provenance")
        group_index = group_labels.index(label)
        row_index = int(np.flatnonzero(row_to_group == group_index)[0])
        sheet_name = f"cluster_{cluster}_samples.png"
        image_name = f"example-{label.removeprefix('brain_')}.png"
        with Image.open(CLUSTERING / sheet_name) as sheet:
            if sheet.size != (2250, 480):
                raise ValueError(f"Unexpected sheet geometry: {sheet_name}")
            crop = sheet.crop(FIRST_PANEL).convert("RGB")
            crops.append((image_name, crop))
        title = LABEL_TITLES[label]
        samples.append(
            {
                "id": label.removeprefix("brain_"),
                "label": label,
                "groupIndex": group_index,
                "title": f"{title} example",
                "description": f"First scan shown in the published cluster {cluster} sheet.",
                "imagePath": f"/research/{image_name}",
                "sheetPath": f"/research/{sheet_name}",
                "sheetCrop": list(FIRST_PANEL),
                "embedding": crop_vectors[row_index].tolist(),
            }
        )
    return samples, crops


def main():
    vectors, scan_ids, labels = load_features()
    archive = json.loads((CLUSTERING / "clustering_metrics.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=2):
        group_ids, row_to_group, group_counts, group_labels, group_vectors = build_groups(
            vectors, scan_ids, labels
        )
        if len(group_ids) != archive["n_grouped_scans"]:
            raise ValueError("Reference group count differs from the archived report")
        crop_vectors = unit_rows(vectors)
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        positions = pca.fit_transform(group_vectors)
        validation = validate_neighbors(
            crop_vectors, row_to_group, labels, group_vectors, group_labels
        )
        samples, crops = build_samples(archive, group_labels, row_to_group, crop_vectors)

    reference = {
        "version": 1,
        "dimension": DIMENSION,
        "embeddingPath": "/research/embeddings.f32",
        "labels": list(LABELS),
        "groups": [
            {
                "id": str(scan_id),
                "label": label,
                "sourceCount": int(count),
                "x": float(position[0]),
                "y": float(position[1]),
            }
            for scan_id, label, count, position
            in zip(group_ids, group_labels, group_counts, positions)
        ],
        "projection": {
            "mean": pca.mean_.tolist(),
            "components": pca.components_.tolist(),
            "explainedVarianceRatio": pca.explained_variance_ratio_.tolist(),
        },
        "samples": samples,
        "validation": validation,
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name, path in FIGURES.items():
        shutil.copyfile(path, PUBLIC / name)
    for filename, crop in crops:
        crop.save(PUBLIC / filename, optimize=True)
        crop.close()
    np.asarray(group_vectors, dtype="<f4").tofile(PUBLIC / "embeddings.f32")
    (PUBLIC / "reference.json").write_text(
        json.dumps(reference, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(group_vectors)} reference groups and {len(samples)} examples. "
        f"Held-out comparison: {validation['correct']}/{validation['count']} crops."
    )


if __name__ == "__main__":
    main()
