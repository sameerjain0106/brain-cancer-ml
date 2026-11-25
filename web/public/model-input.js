import { BGR_CHANNEL_MEANS, MODEL_IMAGE_SIDE, prepareImage } from "./analysis.js";

const panel = document.getElementById("model-input");
const body = document.getElementById("model-input-body");
const distanceFormat = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 5,
  maximumFractionDigits: 5,
});
const changeFormat = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 5,
  maximumFractionDigits: 5,
  signDisplay: "exceptZero",
});
let selection;

function drawPreview(canvas, tensor) {
  const side = MODEL_IMAGE_SIDE;
  const plane = side * side;
  canvas.width = canvas.height = side;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Image preview is unavailable in this browser.");
  const frame = context.createImageData(side, side);
  const [blueMean, greenMean, redMean] = BGR_CHANNEL_MEANS;
  const pixels = frame.data;
  for (let pixel = 0; pixel < plane; pixel += 1) {
    const offset = pixel * 4;
    pixels[offset] = tensor[2 * plane + pixel] + redMean;
    pixels[offset + 1] = tensor[plane + pixel] + greenMean;
    pixels[offset + 2] = tensor[pixel] + blueMean;
    pixels[offset + 3] = 255;
  }
  context.putImageData(frame, 0, 0);
}

async function preparePreview(current) {
  const { src, controller, original, canvas, status } = current;
  original.src = src;
  status.textContent = "Preparing the resized view…";
  try {
    const response = await fetch(src, { signal: controller.signal, credentials: "same-origin" });
    if (!response.ok) throw new Error("The image could not be read.");
    const file = await response.blob();
    if (controller.signal.aborted) return;
    const { url, tensor } = await prepareImage(file);
    URL.revokeObjectURL(url);
    if (controller.signal.aborted) return;
    drawPreview(canvas, tensor);
    current.previewReady = true;
    current.tensor = current.example ? tensor : undefined;
    canvas.hidden = false;
    status.textContent = "";
  } catch (error) {
    if (!controller.signal.aborted) status.textContent = error.message || "The resized view could not load.";
  } finally {
    current.pendingPreview = undefined;
  }
}

function loadPreview(current, needsTensor = false) {
  if (current.previewReady && (!needsTensor || current.tensor)) return;
  current.pendingPreview ??= preparePreview(current);
  return current.pendingPreview;
}

function setReading(row, value) {
  row.querySelector(".model-input-label").textContent = value.label;
  row.querySelector(".model-input-distance").textContent = distanceFormat.format(value.distance);
  row.querySelector(".model-input-range").textContent = value.distant ? "Distant reference match" : "Within the reference cutoff";
  row.dataset.distant = String(value.distant);
  row.hidden = false;
}

function finishReplay(current, message) {
  current.worker?.terminate();
  current.worker = undefined;
  current.replay.disabled = false;
  current.replayStatus.textContent = message;
}

async function replayPreview(current) {
  if (current.replay.disabled) return;
  current.replay.disabled = true;
  current.replayStatus.textContent = "Preparing the published preview…";
  try {
    await loadPreview(current, true);
    if (selection !== current) return;
    if (!current.tensor) throw new Error("The preview could not be prepared. Try again.");
    const worker = new Worker(new URL("./inference-worker.js", import.meta.url));
    current.worker = worker;
    worker.onmessage = ({ data }) => {
      if (selection !== current || current.worker !== worker) return;
      if (data.phase) {
        current.replayStatus.textContent = data.phase === "loading" ? "Loading ResNet50…" : "Comparing the preview on this device…";
        return;
      }
      if (data.error) {
        finishReplay(current, "The comparison could not finish. Try again.");
        return;
      }
      try {
        const replayed = current.compareFeatures(data.features);
        setReading(current.replayed, replayed);
        if (document.activeElement === current.replay) current.replayed.focus();
        current.replay.hidden = true;
        const change = changeFormat.format(replayed.distance - current.distance);
        finishReplay(current, "Distance change " + change + ".");
      } catch {
        finishReplay(current, "The comparison could not finish. Try again.");
      }
    };
    worker.onerror = () => {
      if (selection === current && current.worker === worker) finishReplay(current, "The comparison could not start. Try again.");
    };
    const { tensor } = current;
    current.tensor = undefined;
    worker.postMessage({ id: 1, tensor }, [tensor.buffer]);
  } catch (error) {
    if (selection === current) finishReplay(current, error.message || "The comparison could not start.");
  }
}

function disposeSelection() {
  selection?.controller.abort();
  selection?.worker?.terminate();
  selection = undefined;
}

export function clearModelInput() {
  disposeSelection();
  if (!panel || !body) return;
  panel.hidden = true;
  body.replaceChildren();
}

export function setModelInput(result) {
  if (!panel || !body) return;
  disposeSelection();
  body.innerHTML = `
    <fieldset class="model-input-switch">
      <legend class="sr-only">Image to display</legend>
      <label><input type="radio" name="model-input-view" value="original" checked><span class="model-input-original-label">Original image</span></label>
      <label><input type="radio" name="model-input-view" value="resized"><span>Resized input</span></label>
    </fieldset>
    <div class="model-input-panes" data-view="original">
      <figure class="model-input-original">
        <div class="model-input-image"><img alt="" decoding="async"></div>
        <figcaption>Original image</figcaption>
      </figure>
      <figure class="model-input-resized">
        <div class="model-input-image"><canvas role="img" aria-label="Resized image in RGB" hidden></canvas></div>
        <figcaption>${MODEL_IMAGE_SIDE} × ${MODEL_IMAGE_SIDE} pixels</figcaption>
      </figure>
    </div>
    <p class="model-input-status" role="status"></p>
    <p class="model-input-note">The resized view shows the input pixels in RGB. ResNet50 receives the same pixels in BGR with the channel means subtracted.</p>
    <div class="model-input-example" hidden>
      <p class="model-input-note">The saved match uses features from the original dataset crop. The published preview comes from the study figure. Both comparisons exclude this example’s source group.</p>
      <dl class="model-input-readings">
        <div class="model-input-saved" hidden><dt>Saved match</dt><dd><strong class="model-input-label"></strong><span>Distance <span class="model-input-distance"></span></span><span class="model-input-range"></span></dd></div>
        <div class="model-input-replayed" tabindex="-1" hidden><dt>Reprocessed preview</dt><dd><strong class="model-input-label"></strong><span>Distance <span class="model-input-distance"></span></span><span class="model-input-range"></span></dd></div>
      </dl>
      <button class="model-input-replay" type="button">Reprocess the preview</button>
      <p class="model-input-replay-status" role="status"></p>
    </div>`;
  const current = {
    ...result,
    controller: new AbortController(),
    original: body.querySelector("img"),
    canvas: body.querySelector("canvas"),
    status: body.querySelector(".model-input-status"),
    replay: body.querySelector(".model-input-replay"),
    replayed: body.querySelector(".model-input-replayed"),
    replayStatus: body.querySelector(".model-input-replay-status"),
  };
  selection = current;
  current.original.alt = result.alt;
  if (result.example) {
    body.querySelector(".model-input-original-label").textContent = "Published preview";
    body.querySelector(".model-input-original figcaption").textContent = "Published preview";
    body.querySelector(".model-input-example").hidden = false;
    setReading(body.querySelector(".model-input-saved"), result);
    current.replay.addEventListener("click", () => replayPreview(current));
  }
  const panes = body.querySelector(".model-input-panes");
  body.querySelector(".model-input-switch").addEventListener("change", ({ target }) => {
    panes.dataset.view = target.value;
  });
  panel.hidden = false;
  if (panel.open) void loadPreview(current);
}

panel?.addEventListener("toggle", () => {
  if (panel.open && selection) void loadPreview(selection);
});
