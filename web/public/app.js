import { labels, prepareImage, project, rankGroups, unitVector } from "./analysis.js";

const element = (id) => document.getElementById(id);
const samples = element("samples");
const status = element("status");
const image = element("query-image");
const pick = element("pick-image");
const upload = element("upload");
const matchResult = element("match-result");
const matchTitle = element("comparison-title");
const matchAccent = element("match-accent");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const canvas = element("feature-map");
const mapButtons = Array.from(document.querySelectorAll("[data-map-label]"));
let mapLabel = "all";
let mapBounds;
let mapGroups;
let queryCoordinates = null;
let reference;
let vectors;
let worker;
let pendingUpload = null;
let selectedSample = null;
let selectedUrl = null;
let latestRun = 0;
let matchAnimation;

function say(message, error = false, visible = true) {
  status.classList.toggle("sr-only", !visible);
  status.textContent = message;
  status.dataset.state = error ? "error" : "ready";
}

function showMatch(text, state = "waiting") {
  matchAnimation?.cancel();
  matchAnimation = undefined;
  matchResult.dataset.state = state;
  matchTitle.textContent = text;
  if (state === "match" && !reducedMotion.matches && typeof matchAccent.animate === "function") {
    matchAnimation = matchAccent.animate(
      [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }],
      { duration: 370, easing: "cubic-bezier(.2, .75, .2, 1)" },
    );
  }
}

function localAsset(path) {
  return new URL(path.replace(/^\/+/, ""), import.meta.url);
}

async function loadAsset(path, format) {
  const response = await fetch(localAsset(path));
  if (!response.ok) throw new Error("Could not load the project reference files.");
  return response[format]();
}

function selectMapLabel(label) {
  mapLabel = label;
  for (const button of mapButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.mapLabel === label));
  }
  const selected = label === "all" ? reference.groups : mapGroups.get(label);
  const groups = selected.length.toLocaleString();
  const crops = selected.reduce((total, group) => total + group.sourceCount, 0).toLocaleString();
  element("map-summary").textContent = label === "all"
    ? groups + " groups from " + crops + " MRI crops."
    : groups + " groups from " + crops + " MRI crops labeled " + labels[label].name + ". Other types are faded.";
  drawMap();
}

function configureMap() {
  mapGroups = new Map(Object.keys(labels).map((label) => [
    label, reference.groups.filter((group) => group.label === label),
  ]));
  const bounds = (axis) => {
    const values = reference.groups.map((group) => group[axis]);
    return [Math.min(...values), Math.max(...values)];
  };
  mapBounds = [bounds("x"), bounds("y")];
  for (const button of mapButtons) button.disabled = false;
  selectMapLabel("all");
}

function extendExtent([low, high], query) {
  const space = (high - low) * .23;
  return [Math.min(low, Math.max(low - space, query ?? low)),
    Math.max(high, Math.min(high + space, query ?? high))];
}

function drawPoint(context, x, y, radius, label) {
  context.beginPath();
  if (label === "brain_glioma") {
    context.rect(x - radius * .82, y - radius * .82, radius * 1.64, radius * 1.64);
  } else if (label === "brain_tumor") {
    context.moveTo(x, y - radius * 1.15);
    context.lineTo(x + radius, y + radius * .6);
    context.lineTo(x - radius, y + radius * .6);
    context.closePath();
  } else {
    context.arc(x, y, radius * .86, 0, Math.PI * 2);
  }
  context.fill();
}

function drawMap() {
  const context = canvas.getContext("2d");
  if (!context || !mapBounds) return;
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const bitmapWidth = Math.round(width * ratio);
  const bitmapHeight = Math.round(height * ratio);
  if (canvas.width !== bitmapWidth || canvas.height !== bitmapHeight) {
    canvas.width = bitmapWidth;
    canvas.height = bitmapHeight;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const padding = width > 550 ? 41 : 23;
  const [queryX, queryY] = queryCoordinates ?? [];
  const [minX, maxX] = extendExtent(mapBounds[0], queryX);
  const [minY, maxY] = extendExtent(mapBounds[1], queryY);
  const scale = Math.min((width - 2 * padding) / (maxX - minX), (height - 2 * padding) / (maxY - minY));
  const middleX = (minX + maxX) / 2;
  const middleY = (minY + maxY) / 2;
  const position = (x, y) => [width / 2 + (x - middleX) * scale, height / 2 - (y - middleY) * scale];

  context.lineWidth = 1;
  context.strokeStyle = "#d8e7e3";
  context.beginPath();
  for (const value of [-.2, -.1, 0, .1, .2]) {
    const [x, y] = position(value, value);
    if (x > padding && x < width - padding) {
      context.moveTo(x, padding);
      context.lineTo(x, height - padding);
    }
    if (y > padding && y < height - padding) {
      context.moveTo(padding, y);
      context.lineTo(width - padding, y);
    }
  }
  context.stroke();

  const radius = width > 550 ? 3.45 : 2.6;
  for (const [label, groups] of mapGroups) {
    context.fillStyle = labels[label].color;
    context.globalAlpha = mapLabel === "all" ? .66 : mapLabel === label ? .86 : .13;
    for (const group of groups) {
      const [x, y] = position(group.x, group.y);
      drawPoint(context, x, y, radius, label);
    }
  }
  context.globalAlpha = 1;
  if (!queryCoordinates) return;

  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const [x, y] = position(clamp(queryX, minX, maxX), clamp(queryY, minY, maxY));
  context.beginPath();
  context.moveTo(x, y - 8);
  context.lineTo(x + 8, y);
  context.lineTo(x, y + 8);
  context.lineTo(x - 8, y);
  context.closePath();
  context.lineJoin = "round";
  context.strokeStyle = "#fffdf7";
  context.lineWidth = 5;
  context.stroke();
  context.fillStyle = "#f8b99d";
  context.fill();
  context.strokeStyle = "#723b29";
  context.lineWidth = 1.7;
  context.stroke();
  const outside = queryX < minX || queryX > maxX || queryY < minY || queryY > maxY;
  element("map-caption").textContent = outside
    ? "The selected scan projects outside the map. Its marker sits at the nearest edge."
    : "The two axes use the same scale.";
}

function showImage(url, alt, origin, caption, local = false) {
  if (selectedUrl) URL.revokeObjectURL(selectedUrl);
  selectedUrl = local ? url : null;
  image.src = url;
  image.alt = alt;
  image.hidden = false;
  element("image-origin").textContent = origin;
  element("image-caption").textContent = caption;
}

function markSelection(id) {
  selectedSample = id;
  for (const button of samples.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.sample === id));
  }
}

function renderComparison(values, excludedIndex = -1) {
  const query = unitVector(values, reference.dimension);
  const [best] = rankGroups(query, vectors, reference.groups.length, excludedIndex);
  queryCoordinates = project(query, reference.projection);
  const nearest = labels[reference.groups[best.index].label].name;
  const distant = best.distance > reference.validation.nearestDistance.p95;
  if (distant && excludedIndex === -1) showMatch("Distant reference match", "distant");
  else showMatch(nearest, "match");

  const proximity = element("reference-proximity");
  proximity.hidden = false;
  proximity.dataset.distant = String(distant);
  proximity.textContent = distant
    ? "The closest group is farther away than for 95% of held-out project crops."
    : "This distance is within the range of 95% of held-out project crops.";
  const note = element("comparison-note");
  note.hidden = false;
  note.textContent = excludedIndex !== -1
    ? "The example’s own group is excluded from matching."
    : "The closest match describes resemblance within the study dataset.";
  element("query-key").hidden = false;
  drawMap();
}

function resetComparison(title, note) {
  queryCoordinates = null;
  showMatch(title);
  element("reference-proximity").hidden = true;
  element("query-key").hidden = true;
  const comparisonNote = element("comparison-note");
  comparisonNote.textContent = note;
  comparisonNote.hidden = false;
  element("map-caption").textContent = "The two axes use the same scale.";
  drawMap();
}

function useSample(sample) {
  latestRun += 1;
  if (pendingUpload) {
    worker?.terminate();
    worker = undefined;
    pendingUpload = null;
  }
  pick.disabled = false;
  markSelection(sample.id);
  showImage(localAsset(sample.imagePath).href, "Published MRI project crop labeled " + labels[sample.label].name,
    labels[sample.label].name + " example", "Published MRI crop");
  renderComparison(sample.embedding, sample.groupIndex);
  say("Showing the published " + labels[sample.label].name.toLowerCase() + " example.", false, false);
}

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(localAsset("./inference-worker.js"));
  worker.onmessage = ({ data }) => {
    if (data.id !== latestRun || data.id !== pendingUpload) return;
    if (data.phase === "loading") say("Loading the 47 MB ResNet50 model for the first comparison…");
    else if (data.phase === "computing") say("Extracting ResNet50 features on this device…");
    else {
      pendingUpload = null;
      pick.disabled = false;
      if (data.error) {
        resetComparison("Unavailable", "Try another example or retry with a smaller crop.");
        say("The feature extractor could not run in this browser session.", true);
        return;
      }
      try {
        renderComparison(data.features);
        say("Comparison complete. The image stayed on this device.");
      } catch (error) {
        resetComparison("Unavailable", "Try another crop.");
        say(error.message, true);
      }
    }
  };
  worker.onerror = () => {
    worker?.terminate();
    worker = undefined;
    if (pendingUpload) {
      pendingUpload = null;
      pick.disabled = false;
      resetComparison("Unavailable", "The published examples are still available.");
      say("This browser could not start local model processing.", true);
    }
  };
  return worker;
}

async function useUpload(file) {
  const id = ++latestRun;
  pick.disabled = true;
  say("Preparing the selected image…");
  let url;
  try {
    const prepared = await prepareImage(file);
    url = prepared.url;
    if (latestRun !== id) {
      URL.revokeObjectURL(url);
      return;
    }
    markSelection(null);
    showImage(url, "Locally selected research image", "Selected image",
      "Local file", true);
    resetComparison("In progress", "Comparing the selected crop with saved scan-group features.");
    const activeWorker = ensureWorker();
    pendingUpload = id;
    activeWorker.postMessage({ id, tensor: prepared.tensor }, [prepared.tensor.buffer]);
  } catch (error) {
    if (latestRun !== id) return;
    if (url && selectedUrl !== url) URL.revokeObjectURL(url);
    pick.disabled = false;
    pendingUpload = null;
    if (!selectedSample) resetComparison("Unavailable", "Select a different file or a published example.");
    say(error.message || "The image could not be opened.", true);
  }
}

function renderFollowup(report) {
  const options = [
    ["full", "Full ResNet50 features"],
    ["centered", "Centered ResNet50, all dimensions"],
    ["pca64", "Centered ResNet50, 64 PCA components"],
  ];
  const body = element("followup-table").tBodies[0];
  for (const [key, label] of options) {
    const row = body.insertRow();
    const score = report.results[key];
    row.insertCell().textContent = label;
    row.insertCell().textContent = score.auroc.toFixed(3);
    const accuracy = row.insertCell();
    accuracy.textContent = (100 * score.balancedAccuracy).toFixed(2) + "%";
    const bar = document.createElement("div");
    bar.className = "score-fill";
    bar.setAttribute("aria-hidden", "true");
    bar.style.setProperty("--score", (100 * score.balancedAccuracy) + "%");
    accuracy.append(bar);
  }
  const split = report.split;
  element("followup-description").textContent =
    "A follow-up used the same nearest-cosine scorer and held-out split across all three variants. " +
    "It fit on " + split.trainGliomaGroups + " glioma groups, selected the cutoff on " + split.calibrationGliomaGroups +
    " separate glioma groups, and tested on " + split.testGliomaGroups + " glioma groups plus " + split.testOtherGroups + " other groups.";
  element("followup-note").textContent =
    "Centering alone changed the cosine distances enough to reduce performance. This result is specific to the same-source ResNet50 experiment. The original autoencoder used a different representation, split, and Mahalanobis scoring.";
  element("pca-followup").hidden = false;
}

async function start() {
  loadAsset("research/novelty-benchmark.json", "json").then(renderFollowup).catch(() => {});
  try {
    reference = await loadAsset("research/reference.json", "json");
    vectors = new Float32Array(await loadAsset(reference.embeddingPath, "arrayBuffer"));
    if (reference.version !== 1 || !reference.samples.length ||
        vectors.length !== reference.groups.length * reference.dimension) {
      throw new Error("The saved reference set did not load completely.");
    }
    for (const sample of reference.samples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample-button";
      button.dataset.sample = sample.id;
      button.setAttribute("aria-pressed", "false");
      const thumbnail = document.createElement("img");
      thumbnail.src = localAsset(sample.imagePath).href;
      thumbnail.alt = "";
      button.append(thumbnail, labels[sample.label].name);
      button.addEventListener("click", () => useSample(sample));
      samples.append(button);
    }
    samples.setAttribute("aria-busy", "false");
    configureMap();
    const validation = reference.validation;
    const note = element("reference-validation");
    note.textContent = "When each crop’s entire filename-derived scan group was excluded from matching, the nearest reference label agreed in " +
      validation.correct.toLocaleString() + " of " + validation.count.toLocaleString() + " crops (" +
      (validation.accuracy * 100).toFixed(1) + "%). This is a same-collection check across " + validation.groupCount +
      " inferred groups. Independent patient and scanner performance were outside the study.";
    note.hidden = false;
    useSample(reference.samples.find((sample) => sample.label === "brain_glioma") ?? reference.samples[0]);
  } catch (error) {
    reference = null;
    mapBounds = null;
    for (const button of mapButtons) button.disabled = true;
    element("map-summary").textContent = "The scan groups could not load. Refresh to retry.";
    samples.setAttribute("aria-busy", "false");
    showMatch("Unavailable");
    say("The comparison files could not load. Refresh to retry. The original findings are below.", true);
    console.error(error);
  }
}

for (const button of mapButtons) {
  button.addEventListener("click", () => selectMapLabel(button.dataset.mapLabel));
}
pick.addEventListener("click", () => upload.click());
upload.addEventListener("change", () => {
  const file = upload.files?.[0];
  upload.value = "";
  if (file && reference) useUpload(file);
});
if (typeof ResizeObserver === "function") {
  new ResizeObserver(drawMap).observe(canvas);
} else {
  window.addEventListener("resize", drawMap);
}
start();
