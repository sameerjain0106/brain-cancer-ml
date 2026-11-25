"""Export the project's Keras ResNet50 backbone without a TensorFlow install.

Use the ImageNet no-top weights published with Keras Applications:
https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

Verify the resulting ONNX graph against the committed embeddings and the
original local image crops before writing the browser asset.

Dependencies: h5py numpy onnx onnxruntime pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

WEB = Path(__file__).resolve().parents[1]
REFERENCE = WEB.parent / "outputs" / "clustering"
WEIGHTS_SHA256 = "66c8b43daff3fcc15bc4f30e3d2a167e21a14d9c9598a5394e5516471f4af504"
BACKBONE_INPUT = "image"
BACKBONE_OUTPUT = "features"
IMAGE_SIZE = 512
FEATURE_SIZE = 2048
BGR_MEANS = np.array([103.939, 116.779, 123.68], dtype=np.float32)
BATCH_NORM_EPSILON = 1.001e-5
MIN_COSINE_SIMILARITY = 0.99999
MAX_RELATIVE_ERROR = 0.005


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_graph(weights: h5py.File) -> onnx.ModelProto:
    # Keras v1 stem, projection and 3-4-6-3 bottleneck layout. Strides belong
    # to each stage's first 1x1 convolution. Keras HDF5 kernels are HWIO.
    nodes = []
    tensors = []

    def node(kind: str, name: str, inputs: list[str], **attributes) -> str:
        nodes.append(helper.make_node(kind, inputs, [name], name=name, **attributes))
        return name

    def array(layer: str, value: str) -> np.ndarray:
        return np.asarray(weights[f"{layer}/{layer}/{value}:0"], dtype=np.float32)

    def conv(
        previous: str, layer: str, *, stride: int = 1, padding: int = 0
    ) -> str:
        conv_layer = f"{layer}_conv"
        norm_layer = f"{layer}_bn"
        kernel = array(conv_layer, "kernel").transpose((3, 2, 0, 1))
        bias = array(conv_layer, "bias")
        # Fold batch normalization into the frozen convolution. Store weights
        # in float16, and use float32 activation and convolution in browsers.
        scale = array(norm_layer, "gamma") / np.sqrt(
            array(norm_layer, "moving_variance") + BATCH_NORM_EPSILON
        )
        folded_kernel = (kernel * scale[:, None, None, None]).astype(np.float16)
        folded_bias = (bias - array(norm_layer, "moving_mean")) * scale
        folded_bias += array(norm_layer, "beta")

        kernel_name = f"{layer}_kernel_half"
        bias_name = f"{layer}_bias"
        tensors.append(numpy_helper.from_array(folded_kernel, kernel_name))
        tensors.append(numpy_helper.from_array(folded_bias, bias_name))
        float_kernel = node("Cast", f"{layer}_kernel", [kernel_name], to=TensorProto.FLOAT)
        return node(
            "Conv", conv_layer, [previous, float_kernel, bias_name],
            pads=[padding] * 4, strides=[stride] * 2,
        )

    stem = conv(BACKBONE_INPUT, "conv1", stride=2, padding=3)
    stem = node("Relu", "conv1_relu", [stem])
    # Keras explicitly zero-pads between the ReLU and pool. The activations
    # are nonnegative here, so ONNX MaxPool padding yields the same values.
    output = node(
        "MaxPool", "pool1_pool", [stem],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2],
    )

    for stage, block_count in enumerate((3, 4, 6, 3), start=2):
        for block in range(1, block_count + 1):
            name = f"conv{stage}_block{block}"
            stride = 2 if stage > 2 and block == 1 else 1
            shortcut = conv(output, f"{name}_0", stride=stride) if block == 1 else output
            main = conv(output, f"{name}_1", stride=stride)
            main = node("Relu", f"{name}_1_relu", [main])
            main = conv(main, f"{name}_2", padding=1)
            main = node("Relu", f"{name}_2_relu", [main])
            main = conv(main, f"{name}_3")
            merged = node("Add", f"{name}_add", [shortcut, main])
            output = node("Relu", f"{name}_out", [merged])

    pooled = node("GlobalAveragePool", "avg_pool", [output])
    node("Flatten", BACKBONE_OUTPUT, [pooled], axis=1)
    graph = helper.make_graph(
        nodes,
        "Keras ResNet50 ImageNet embedding",
        [helper.make_tensor_value_info(
            BACKBONE_INPUT, TensorProto.FLOAT, [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        )],
        [helper.make_tensor_value_info(
            BACKBONE_OUTPUT, TensorProto.FLOAT, [1, FEATURE_SIZE],
        )],
        initializer=tensors,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)],
        ir_version=8, producer_name="brain-cancer-ml",
    )
    onnx.checker.check_model(model)
    return model


def image_to_input(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        pixels = source.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST,
        )
        bgr = np.asarray(pixels, dtype=np.float32)[:, :, ::-1] - BGR_MEANS
    return np.ascontiguousarray(bgr.transpose((2, 0, 1))[None])


def verify_parity(model: bytes, raw: Path, reference: Path) -> dict:
    names = np.load(reference / "file_names.npy", allow_pickle=False)
    labels = np.load(reference / "true_labels.npy", allow_pickle=False)
    group_ids = np.load(reference / "image_ids.npy", allow_pickle=False)
    expected = np.load(reference / "features.npy", mmap_mode="r", allow_pickle=False)
    if expected.shape != (len(names), FEATURE_SIZE) or len(labels) != len(names):
        raise ValueError("Local metadata and reference embeddings do not align")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    measurements = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        # Check three different scan groups per source class.
        _, positions = np.unique(group_ids[rows], return_index=True)
        anchors = np.linspace(0, len(positions) - 1, 3, dtype=int)
        for row in rows[positions[anchors]]:
            folder, filename = names[row]
            path = raw / str(folder) / str(filename)
            original = expected[row]
            start = time.perf_counter()
            output = session.run(
                [BACKBONE_OUTPUT], {BACKBONE_INPUT: image_to_input(path)}
            )[0][0]
            output_norm = np.linalg.norm(output)
            expected_norm = np.linalg.norm(original)
            cosine = float(np.dot(output, original) / (output_norm * expected_norm))
            error = float(np.linalg.norm(output - original) / expected_norm)
            if not np.isfinite(cosine + error) or (
                cosine < MIN_COSINE_SIMILARITY or error > MAX_RELATIVE_ERROR
            ):
                raise ValueError(
                    f"ONNX parity failed in {label} at reference row {row}: "
                    f"cosine={cosine:.8f}, relative error={error:.8f}"
                )
            measurements.append((cosine, error, time.perf_counter() - start))

    minimum_cosine = min(measurement[0] for measurement in measurements)
    maximum_error = max(measurement[1] for measurement in measurements)
    print(f"Maximum local CPU verification time: {max(row[2] for row in measurements):.2f} s")
    return {
        "samples": len(measurements),
        "minCosineSimilarity": minimum_cosine,
        "maxRelativeL2Error": maximum_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="Local original MRI folders")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=WEB / "public" / "model" / "resnet50.onnx")
    options = parser.parse_args()
    weights_hash = sha256_file(options.weights)
    if weights_hash != WEIGHTS_SHA256:
        raise ValueError("Weights do not match the expected Keras ImageNet no-top file")
    with h5py.File(options.weights, "r") as weights:
        payload = build_graph(weights).SerializeToString()
    parity = verify_parity(payload, options.raw, options.reference)
    manifest = {
        "source": "Keras ResNet50 ImageNet no-top weights",
        "kerasWeightSha256": weights_hash,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "input": BACKBONE_INPUT,
        "inputLayout": "NCHW",
        "inputShape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "preprocessing": "RGB nearest-neighbor resize, then BGR with channel means subtracted",
        "output": BACKBONE_OUTPUT,
        "outputShape": [1, FEATURE_SIZE],
        "parityAgainstSavedEmbeddings": parity,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_bytes(payload)
    options.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
