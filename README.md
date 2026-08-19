# Brain Tumor MRI — Finalized Project

Three approaches to the same brain tumor MRI dataset (glioma / meningioma /
tumor), reviewed, fixed, executed against the real dataset, and documented.
This README explains what changed, why, and what's still open.

---

## TL;DR — what changed and what it found

| Approach | Key fix | Real result |
|---|---|---|
| **Supervised classification** | Path portability only; core logic untouched | Random Forest (900 trees): **72.7% accuracy**, 0.729 F1 — unchanged from original run, outputs preserved |
| **Unsupervised clustering** (ResNet50 + K-Means) | Fixed a variable-name collision that silently destroyed ground-truth labels; added the quantitative evaluation that was missing | **ARI = 1.0, purity = 1.0** on ~600 grouped scans |
| **Anomaly detection** (autoencoder + Mahalanobis) | Fixed train/test leakage (model was being evaluated on data it trained on) | **50% accuracy** on the corrected, leak-free evaluation — a real, honest finding |

---

## 1. Directory structure

```
brain-cancer-ml/
├── supervised_classification.ipynb   # PCA + LogReg/KNN/RF (finalized)
├── clustering_analysis.ipynb         # ResNet50 + K-Means (finalized, executed)
├── anomaly_detection.ipynb           # Autoencoder + Mahalanobis (finalized, executed)
│
├── extract_features.py            # ResNet50 feature extraction (batched, resumable)
├── clustering_analysis.py         # Script version of the clustering notebook
├── anomaly_detection.py           # Script version of the anomaly detection notebook
│
├── outputs/
│   ├── clustering/                # features.npy, metrics, figures
│   └── anomaly_detection/         # metrics, figures
│
├── originals/                     # Your original files, untouched
│   ├── project_code_sami_original.ipynb
│   ├── KMeans_original.ipynb
│   └── AD_original.py
│
├── requirements.txt
└── README.md                      # this file
```

**Note:** The raw image folders (`brain_glioma/`, `brain_menin/`, `brain_tumor/`, containing ~6,000 total images) are **not included** in this repository due to size constraints. You will need to obtain the original dataset and place these folders in the repository root for the notebooks and scripts to run.

All scripts/notebooks resolve paths relative to their own location
(`Path.cwd()` / `Path(__file__).parent`), so this works as long as the
three class image folders are placed in the same directory as the notebooks and scripts.

---

## 2. What I changed, and why

### Supervised classification (`supervised_classification.ipynb`)

**Kept unchanged:** every modeling decision — PCA to 1,000 components,
the exact train/test split, Logistic Regression / KNN / Random Forest
with their original hyperparameters. All existing outputs (confusion
matrices, accuracy-vs-k and accuracy-vs-n_estimators curves, classification
reports) are **preserved from your original run**, not regenerated.

**Changed:** only the image-loading paths, from bare relative filenames
to portable paths that expect the `brain_glioma/`, `brain_menin/`, and `brain_tumor/` 
folders to be present in the repository root.

**Why I didn't re-run it:** the raw-pixel approach loads all ~6,000
images as flattened float32 vectors (262,144 features each) into one
in-memory matrix before running PCA — roughly 6GB+ of RAM. That exceeded
what was available in my execution environment. Since the only code
change was the path structure (not the modeling logic, data selection, or
sample count), the existing outputs should still be accurate — but this
should be verified by re-running the notebook top-to-bottom on a machine
with sufficient RAM.

**Found but not fixed:** the loading loops use `range(1, 2005)`, i.e.
exactly 2,004 images per class. The actual raw dataset has 2,048 tumor images
— 44 more than the notebook loads. I flagged this in the notebook rather
than silently fixing it, because correcting it would change the PCA fit
and invalidate the existing (preserved) outputs.

### Unsupervised clustering (`clustering_analysis.ipynb`)

**Kept unchanged:** ResNet50 (ImageNet weights, avg-pooled, no top),
loaded at 512×512 (not downsampled), K-Means with k=3 chosen via the
elbow method — same methodology as your original notebook.

**Fixed — variable collision:** the original notebook did
`labels = kmeans.fit_predict(...)`, silently overwriting the array that
held the true class labels from the file loader. This is why no
quantitative evaluation existed in the original — the ground truth was
gone by the time clustering finished. This version keeps
`grouped_true_labels` and `cluster_assignments` as separate, clearly
named arrays throughout.

**Fixed — memory usage:** the original loaded all 6,056 images into one
array before running the model (~5GB+ as float32). This version extracts
features in small batches (`extract_features.py`), holding at most one
batch of raw pixels in memory at a time — the same methodology, applied
in a way that scales.

**Added — the actual evaluation:** Adjusted Rand Index and a
Hungarian-algorithm-aligned cluster-vs-true-label confusion matrix.
Result: **ARI = 1.0, purity = 1.0 (603/603 grouped scans)**. This
confirms, with a real metric, what your manual check (tracing all 201
images per cluster back to their source folder) had already found.

**Read this result with the right scope** (also noted in the notebook):
this shows ImageNet-pretrained visual features are highly discriminative
on *this specific, curated, single-source dataset* — not that the
pipeline is a validated diagnostic tool. It hasn't been tested for
stability across seeds, and perfect separation on ~600 scans from one
source doesn't establish generalization to new hospitals or scanners.

### Anomaly detection (`anomaly_detection.ipynb`)

**Kept unchanged:** the 3-layer convolutional autoencoder architecture,
128×128 grayscale preprocessing, 10 training epochs, PCA to 100 latent
components, Mahalanobis distance, 90th-percentile threshold — all as
originally written.

**Fixed — train/test leakage:** the original trained on glioma images
and then evaluated on data from the same source folder, meaning the "test" set
included the exact images used for training. This version splits the glioma
images 80/20 **before** training (1,604 train / 400 held out); the
autoencoder never sees the held-out 400. Evaluation runs on those 400 plus
all meningioma and tumor images.

**Fixed — fragile path/class discovery:** the original used
`torchvision.ImageFolder`, which auto-discovers *every* subdirectory as
a class. This breaks the moment any non-class folder (like `outputs/`)
sits next to the data. Replaced with an explicit loader that only reads
the three named class folders.

**Result, honestly reported:** once the leak is closed, performance
drops to **50% overall accuracy** (0.14 precision / 0.89 recall on
glioma-like, 0.98 precision / 0.46 recall on anomaly) — close to chance,
with a strong bias toward calling things "glioma-like." This is a
legitimate finding, not a failure to hide: a single Gaussian over a
100-dim PCA of a small autoencoder's latent space isn't a strong enough
anomaly detector to separate glioma from other tumor types here. It's
also a useful, honest contrast against the much stronger supervised and
clustering results on the same underlying data.

---

## 3. Reproducing this yourself

First, **obtain the original brain tumor MRI dataset and place the three class folders
(`brain_glioma/`, `brain_menin/`, `brain_tumor/`) in the repository root.**

Then:

```bash
pip install -r requirements.txt

# 1. Extract ResNet50 features (slow on CPU-only machines — a GPU or
#    multi-core machine will be much faster)
python extract_features.py

# 2. Run the clustering analysis (fast — seconds, once features exist)
jupyter nbconvert --to notebook --execute clustering_analysis.ipynb

# 3. Run anomaly detection (a few minutes on CPU)
jupyter nbconvert --to notebook --execute anomaly_detection.ipynb

# 4. Supervised classification needs ~8GB+ RAM available
jupyter nbconvert --to notebook --execute supervised_classification.ipynb
```

All three notebooks use `SEED = 42` (or equivalent) wherever randomness
is involved, so results should reproduce exactly on the same dataset.

---

## 4. What to verify manually

1. **Re-run `supervised_classification.ipynb` end-to-end** on a machine
   with enough RAM, to confirm the preserved outputs still match — I
   could not do this in my environment (see above).
2. **Decide on the 2,004-vs-2,048 tumor image count** — leave as-is to
   match the existing results, or extend the loop and re-run everything
   downstream of that cell.
3. **Sanity-check the "perfect separation" scope note** in
   `clustering_analysis.ipynb` reads the way you want it to for a
   portfolio audience — it's a genuinely strong result and I wanted to
   make sure the framing doesn't overclaim.
4. **`anomaly_detection.ipynb`'s weak result** is real and reproducible
   with the given seed, but if you have domain intuition for why it
   underperforms (beyond what's noted in the notebook's Limitations
   section), worth adding as your own commentary.

---

## 5. Environment notes (for context, not action needed)

This review ran in a constrained sandbox: 1 CPU core, ~4GB RAM, no GPU.
Two infrastructure issues came up and were resolved:
- The default Keras ResNet50 weights download (`storage.googleapis.com`)
  was unreachable; verified, hash-matched weights were sourced from
  `keras-team`'s GitHub release instead.
- PyTorch/TensorFlow's default pip installs bundle several GB of CUDA
  libraries even for CPU-only use; installed with `--no-deps` plus the
  minimal pure-Python dependencies actually needed at runtime.

Neither of these should matter on a normal machine with regular internet
access, but they're recorded here in case you hit either while
reproducing this locally.
