import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(
    root="/Users/diyamannacherry/Desktop/4600/Brain_Cancer/",
    transform=transform
)
glioma_idx = dataset.class_to_idx['brain_glioma']
dataset.samples = [s for s in dataset.samples if s[1] == glioma_idx]
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 16x16
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        outputs, _ = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
model.eval()
latent_vectors = []
with torch.no_grad():
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        _, z = model(imgs)
        z = torch.flatten(z, start_dim=1)  # flatten (16384)
        latent_vectors.append(z.cpu())
latent_vectors = torch.cat(latent_vectors).numpy()
pca = PCA(n_components=100)
latent_vectors = pca.fit_transform(latent_vectors)
mu = np.mean(latent_vectors, axis=0)
cov = np.cov(latent_vectors, rowvar=False)
cov += np.eye(cov.shape[0]) * 1e-5
cov_inv = np.linalg.inv(cov)
test_dataset = datasets.ImageFolder(
    root="/Users/diyamannacherry/Desktop/4600/Brain_Cancer/",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
distances = []
labels = []
with torch.no_grad():
    for imgs, labels_batch in test_loader:
        imgs = imgs.to(device)
        _, z = model(imgs)
        z = torch.flatten(z, start_dim=1).cpu().numpy()
        z = pca.transform(z)
        for i in range(z.shape[0]):
            diff = z[i] - mu
            d = np.sqrt(diff @ cov_inv @ diff.T)
            distances.append(d)
            labels.append(labels_batch[i].item())
train_distances = []
for z in latent_vectors:
    diff = z - mu
    d = np.sqrt(diff @ cov_inv @ diff.T)
    train_distances.append(d)
threshold = np.percentile(train_distances, 90)
preds = [1 if d > threshold else 0 for d in distances]

plt.figure()
plt.hist(train_distances, bins=50, alpha=0.6, label="Train (Glioma)")
plt.hist(distances, bins=50, alpha=0.6, label="Test (All)")
plt.axvline(threshold, color='red', linestyle='--', label="Threshold")
plt.legend()
plt.title("Mahalanobis Distance Comparison")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

print("\nThreshold:", threshold)
print("Total samples:", len(distances))
print("Detected anomalies:", sum(preds))

y_true_binary = np.array([1 if lbl == glioma_idx else 0 for lbl in labels])
y_pred_binary = np.array(preds)

cm = confusion_matrix(y_true_binary, y_pred_binary)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Anomaly (0)", "Glioma (1)"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Binary Confusion Matrix: Glioma vs Anomaly")
plt.show()

print(classification_report(
    y_true_binary,
    y_pred_binary,
    target_names=["Anomaly", "Glioma"]
))
