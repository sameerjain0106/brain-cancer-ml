let sessionPromise;
let tasks = Promise.resolve();

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (!self.ort) importScripts("./vendor/ort.wasm.min.js");
      self.ort.env.wasm.numThreads = 1;
      self.ort.env.wasm.proxy = false;
      self.ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
      return self.ort.InferenceSession.create("./model/resnet50.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })().catch((error) => {
      sessionPromise = undefined;
      throw error;
    });
  }
  return sessionPromise;
}

async function compare(id, pixels) {
  try {
    if (!sessionPromise) self.postMessage({ id, phase: "loading" });
    const session = await getSession();
    self.postMessage({ id, phase: "computing" });
    const image = new self.ort.Tensor("float32", pixels, [1, 3, 512, 512]);
    const result = await session.run({ image });
    const features = Float32Array.from(result.features.data);
    self.postMessage({ id, features }, [features.buffer]);
  } catch (error) {
    console.error("MRI feature extraction failed", error);
    self.postMessage({ id, error: true });
  }
}

self.onmessage = ({ data }) => {
  tasks = tasks.then(() => compare(data.id, data.tensor));
};
