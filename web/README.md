# Brain tumor MRI research site

The project explorer compares JPG and PNG research crops with the ResNet50 references published by the original project. Switch between projected 2D distance and full-feature cosine distance to see how the nearest groups change. The browser processes a selected image locally. The site has no upload endpoint, storage service, or analytics. Interactive results describe proximity to groups in the source dataset. Use de-identified research examples.

## Run locally

From this directory, serve the public folder.

```powershell
python -m http.server 4173 --bind 127.0.0.1 --directory public
```

Open http://127.0.0.1:4173. The three published examples use their saved vectors and load without the ONNX model. The browser fetches the 47 MB feature extractor only when a visitor selects an image. First-time model loading and inference can take longer on a phone.

## Deploy

Import the existing brain-cancer-ml GitHub repository in Vercel and select `web` as the root directory. The checked-in Vercel configuration serves the static `public` directory. The deployment needs no API keys, databases, or server functions. Keep the ONNX model and the ONNX Runtime Web assets on the same origin. Image comparison runs locally after these assets load.

## Reproduce the assets and comparisons

- `tools/build-reference-data.py` reads the published ResNet50 embeddings and metadata under `../outputs/clustering/`. It writes the normalized 603-group index, feature map, published example crops, and same-collection validation. The groups come from class folders and filenames.
- `tools/benchmark-novelty.py` compares full, centered, and 64-component PCA ResNet50 embeddings using the same held-out groups and cosine novelty scorer. It is a new retrospective comparison separate from the original autoencoder experiment.
- `tools/export-resnet.py` exports the original ImageNet ResNet50 backbone from official Keras weights, using a browser-compatible ONNX representation. See the script's `--help` and `public/model/resnet50.json` for input, preprocessing, dependencies, and parity details. Use local original data only to validate the export.
- The browser runtime is ONNX Runtime Web 1.20.1, distributed with its license in `public/vendor/`.

Raw dataset files and personal scans belong outside the web directory. The sample thumbnails derive from the research figures already published in this repository.
