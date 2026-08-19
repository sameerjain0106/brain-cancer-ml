"""
anomaly_detection.py

Convolutional-autoencoder + Mahalanobis-distance anomaly detector for
glioma MRI scans. This is a fixed/finalized version of the original
AD.py, with the following corrections and improvements over the
original (see README.md for the full explanation):

1. TRAIN/TEST LEAKAGE FIX
   The original script trained the autoencoder on glioma images, then
   evaluated on an ImageFolder built from the *same* root directory --
   meaning the "test" set included the exact glioma images the model
   was trained on. This version does a proper 80/20 split of the
   glioma class *before* training: the autoencoder only ever sees the
   training split, and evaluation uses the held-out glioma split plus
   the full menin/tumor classes (as anomalies).

2. PORTABLE, EXPLICIT DATA ACCESS
   Original used a hardcoded absolute path
   ("/Users/diyamannacherry/Desktop/4600/Brain_Cancer/") and
   torchvision's ImageFolder, which auto-discovers every subdirectory
   as a class -- meaning it would break if any non-class folder
   (e.g. an outputs/ directory) is placed alongside the data. This
   version resolves the data directory relative to this file and
   only ever looks at the three named class folders.

3. REPRODUCIBILITY
   Fixed random seeds for the train/test split and model init.

4. OUTPUTS SAVED TO DISK
   All figures and metrics are saved to outputs/anomaly_detection/
   instead of only being shown inline.

The core methodology is unchanged: grayscale 128x128 images, a 3-layer
convolutional autoencoder trained on glioma scans only, PCA on the
flattened latent space, Mahalanobis distance against the training
distribution, and a 90th-percentile threshold.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
CLASSES = ["brain_glioma", "brain_menin", "brain_tumor"]
TARGET_CLASS = "brain_glioma"  # the class the autoencoder is trained to model
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
LATENT_PCA_COMPONENTS = 100
TEST_SPLIT = 0.2
THRESHOLD_PERCENTILE = 90
OUTPUT_DIR = BASE_DIR / "outputs" / "anomaly_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------------
# Explicit dataset — only looks at the three known class folders,
# unlike ImageFolder's auto-discovery of every subdirectory.
# ------------------------------------------------------------------
class BrainScanDataset(Dataset):
    def __init__(self, file_list, transform):
        # file_list: list of (path, class_name) tuples
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, cls = self.file_list[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, cls


def list_class_files(cls):
    folder = BASE_DIR / cls
    exts = (".jpg", ".jpeg", ".png")
    return sorted([folder / f for f in folder.iterdir() if f.suffix.lower() in exts])


# ------------------------------------------------------------------
# Model — unchanged from the original
# ------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def mahalanobis_distances(Z, mu, cov_inv):
    diffs = Z - mu
    # (Z - mu) @ cov_inv @ (Z - mu).T, diagonal only, vectorized
    left = diffs @ cov_inv
    d2 = np.einsum("ij,ij->i", left, diffs)
    return np.sqrt(np.clip(d2, 0, None))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.Resize(IMG_SIZE), transforms.Grayscale(), transforms.ToTensor()]
    )

    # ---- Build file lists per class ----
    files_by_class = {cls: list_class_files(cls) for cls in CLASSES}
    for cls, files in files_by_class.items():
        print(f"  {cls}: {len(files)} images")

    # ---- Split the target class into train/test (fixes leakage) ----
    target_files = files_by_class[TARGET_CLASS]
    rng = random.Random(SEED)
    shuffled = target_files.copy()
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * TEST_SPLIT)
    glioma_test_files = shuffled[:n_test]
    glioma_train_files = shuffled[n_test:]
    print(f"\n{TARGET_CLASS}: {len(glioma_train_files)} train / {len(glioma_test_files)} held-out test")

    train_list = [(p, TARGET_CLASS) for p in glioma_train_files]
    train_dataset = BrainScanDataset(train_list, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Evaluation set: held-out glioma (normal) + all other classes (anomalies)
    eval_list = [(p, TARGET_CLASS) for p in glioma_test_files]
    for cls in CLASSES:
        if cls != TARGET_CLASS:
            eval_list += [(p, cls) for p in files_by_class[cls]]
    eval_dataset = BrainScanDataset(eval_list, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Evaluation set: {len(eval_list)} images "
          f"({len(glioma_test_files)} held-out glioma + "
          f"{len(eval_list) - len(glioma_test_files)} from other classes)")

    # ---- Train autoencoder on glioma-train only ----
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            outputs, _ = model(imgs)
            loss = criterion(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Training loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE Loss")
    plt.title(f"Autoencoder Training Loss ({TARGET_CLASS} only, n={len(glioma_train_files)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close()

    # ---- Extract latent vectors for TRAIN set, fit PCA + Gaussian ----
    model.eval()
    train_latents = []
    with torch.no_grad():
        for imgs, _ in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False):
            imgs = imgs.to(device)
            _, z = model(imgs)
            z = torch.flatten(z, start_dim=1)
            train_latents.append(z.cpu())
    train_latents = torch.cat(train_latents).numpy()

    pca = PCA(n_components=LATENT_PCA_COMPONENTS, random_state=SEED)
    train_latents_pca = pca.fit_transform(train_latents)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"\nLatent PCA ({LATENT_PCA_COMPONENTS} components): "
          f"{explained_var*100:.1f}% variance explained")

    mu = np.mean(train_latents_pca, axis=0)
    cov = np.cov(train_latents_pca, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-5
    cov_inv = np.linalg.inv(cov)

    train_distances = mahalanobis_distances(train_latents_pca, mu, cov_inv)
    threshold = np.percentile(train_distances, THRESHOLD_PERCENTILE)
    print(f"Anomaly threshold ({THRESHOLD_PERCENTILE}th percentile of train distances): {threshold:.3f}")

    # ---- Extract latent vectors for EVAL set (held-out glioma + others) ----
    eval_latents = []
    eval_true_class = []
    with torch.no_grad():
        for imgs, classes in eval_loader:
            imgs = imgs.to(device)
            _, z = model(imgs)
            z = torch.flatten(z, start_dim=1).cpu().numpy()
            eval_latents.append(z)
            eval_true_class.extend(classes)
    eval_latents = np.concatenate(eval_latents, axis=0)
    eval_latents_pca = pca.transform(eval_latents)
    eval_distances = mahalanobis_distances(eval_latents_pca, mu, cov_inv)

    preds = (eval_distances > threshold).astype(int)  # 1 = anomaly, 0 = normal (glioma-like)
    y_true_binary = np.array([0 if c == TARGET_CLASS else 1 for c in eval_true_class])

    # ---- Distance distribution plot ----
    plt.figure(figsize=(8, 5))
    plt.hist(train_distances, bins=50, alpha=0.6, label=f"Train ({TARGET_CLASS})", color="#4DD8C4")
    plt.hist(eval_distances, bins=50, alpha=0.6, label="Eval (held-out glioma + other classes)", color="#D9A45B")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold (p{THRESHOLD_PERCENTILE})")
    plt.legend()
    plt.title("Mahalanobis Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_distribution.png", dpi=150)
    plt.close()

    # ---- Confusion matrix + report ----
    cm = confusion_matrix(y_true_binary, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Glioma-like (0)", "Anomaly (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Anomaly Detection: Glioma-like vs. Anomaly (held-out evaluation set)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    report = classification_report(
        y_true_binary, preds, target_names=["Glioma-like", "Anomaly"], output_dict=True
    )
    report_text = classification_report(y_true_binary, preds, target_names=["Glioma-like", "Anomaly"])
    print("\n" + report_text)

    # ---- Save metrics ----
    metrics = {
        "target_class": TARGET_CLASS,
        "train_size": len(glioma_train_files),
        "eval_size": len(eval_list),
        "eval_held_out_glioma": len(glioma_test_files),
        "epochs": EPOCHS,
        "final_train_loss": train_losses[-1],
        "latent_pca_components": LATENT_PCA_COMPONENTS,
        "latent_pca_explained_variance": float(explained_var),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_value": float(threshold),
        "confusion_matrix": {
            "labels": ["Glioma-like (0)", "Anomaly (1)"],
            "matrix": cm.tolist(),
        },
        "classification_report": report,
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    main()
