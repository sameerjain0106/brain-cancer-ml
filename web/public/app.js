import { closestGroups, labels, prepareImage, project, unitVector } from "./analysis.js";

const element = (id) => document.getElementById(id);
const samples = element("samples");
const status = element("status");
const image = element("query-image");
const pick = element("pick-image");
const upload = element("upload");
const canvas = element("feature-map");
let reference;
let vectors;
let worker;
let pendingUpload = null;
let selectedSample = null;
let selectedUrl = null;
let latestRun = 0;
let comparison = null;
let pinnedGroup = null;
let hoveredGroup = null;
let chartPositions = [];
let hoverFrame = 0;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

function say(message, error = false) {
  status.textContent = message;
  status.dataset.state = error ? "error" : "ready";
}

function localAsset(path) {
  return new URL(path.replace(/^\/+/, ""), import.meta.url);
}

async function loadAsset(path, format) {
  const response = await fetch(localAsset(path));
  if (!response.ok) throw new Error("Could not load the project reference files.");
  return response[format]();
}

function swatch(label, className = "") {
  const dot = document.createElement("i");
  dot.className = "swatch " + className;
  if (label) dot.style.setProperty("--swatch", labels[label].color);
  dot.setAttribute("aria-hidden", "true");
  return dot;
}

function drawMap() {
  const context = canvas.getContext("2d");
  if (!context) return;
  const { width, height } = canvas.getBoundingClientRect();
  if (width <= 0 || height <= 0) return;
  const pixelRatio = Math.min(devicePixelRatio || 1, 2);
  const bitmapWidth = Math.round(width * pixelRatio);
  const bitmapHeight = Math.round(height * pixelRatio);
  if (canvas.width !== bitmapWidth || canvas.height !== bitmapHeight) {
    canvas.width = bitmapWidth;
    canvas.height = bitmapHeight;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  const padding = 22;
  context.strokeStyle = "#d1d5e8";
  context.beginPath();
  for (let step = 0; step <= 5; step += 1) {
    const x = padding + (width - 2 * padding) * step / 5;
    const y = padding + (height - 2 * padding) * step / 5;
    context.moveTo(x, padding);
    context.lineTo(x, height - padding);
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
  }
  context.stroke();
  if (!reference) return;
  const groups = reference.groups;
  const xs = groups.map((group) => group.x);
  const ys = groups.map((group) => group.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const limit = (value) => Math.max(0, Math.min(1, value));
  const position = (x, y) => [
    padding + limit((x - minX) / (maxX - minX)) * (width - 2 * padding),
    height - padding - limit((y - minY) / (maxY - minY)) * (height - 2 * padding),
  ];
  chartPositions = groups.map((group) => position(group.x, group.y));
  context.globalAlpha = 0.62;
  for (let index = 0; index < groups.length; index += 1) {
    const group = groups[index];
    const [x, y] = chartPositions[index];
    context.fillStyle = labels[group.label].color;
    context.beginPath();
    context.arc(x, y, 2.5, 0, Math.PI * 2);
    context.fill();
  }
  context.globalAlpha = 1;
  const inspectedIndex = hoveredGroup ?? pinnedGroup;
  if (inspectedIndex !== null && chartPositions[inspectedIndex]) {
    const [x, y] = chartPositions[inspectedIndex];
    context.strokeStyle = "#151a56";
    context.beginPath();
    context.moveTo(x, padding);
    context.lineTo(x, height - padding);
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
    context.globalAlpha = 0.3;
    context.stroke();
    context.globalAlpha = 1;
    context.lineWidth = 2;
    context.beginPath();
    context.arc(x, y, 7, 0, Math.PI * 2);
    context.stroke();
    context.lineWidth = 1;
  }
  if (!comparison) return;
  const [queryX, queryY] = comparison.coordinates;
  const [qx, qy] = position(queryX, queryY);
  for (const match of comparison.matches) {
    const group = groups[match.index];
    const [gx, gy] = position(group.x, group.y);
    context.setLineDash([3, 4]);
    context.strokeStyle = "#2935699d";
    context.beginPath();
    context.moveTo(qx, qy);
    context.lineTo(gx, gy);
    context.stroke();
    context.setLineDash([]);
    context.strokeStyle = labels[group.label].color;
    context.beginPath();
    context.arc(gx, gy, 5.5, 0, Math.PI * 2);
    context.stroke();
  }
  context.strokeStyle = "#ffffff";
  context.lineWidth = 2;
  context.fillStyle = "#151a56";
  context.beginPath();
  context.arc(qx, qy, 6, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.lineWidth = 1;
  const outside = queryX < minX || queryX > maxX || queryY < minY || queryY > maxY;
  element("map-caption").textContent = outside
    ? "The query lies beyond the plotted reference span. Its marker sits on the plot boundary."
    : "Two dimensions for the plot. All 2,048 features for the comparison.";
}

function showImage(url, alt, origin, caption, local = false) {
  if (selectedUrl) URL.revokeObjectURL(selectedUrl);
  selectedUrl = local ? url : null;
  image.src = url;
  image.alt = alt;
  image.hidden = false;
  element("scan-empty").hidden = true;
  element("image-origin").textContent = origin;
  element("image-caption").textContent = caption;
}

function markSelection(id) {
  selectedSample = id;
  for (const button of samples.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.sample === id));
  }
}

function inspectGroup() {
  const index = hoveredGroup ?? pinnedGroup;
  const group = index === null ? null : reference?.groups[index];
  const note = element("map-inspection");
  note.textContent = group
    ? labels[group.label].name + " · group " + (String(group.id).match(/\d+$/)?.[0] ?? group.id) + " · " + group.sourceCount + " crops"
    : "Select a point or choose a match below.";
  note.dataset.inspecting = String(Boolean(group));
  drawMap();
}

function frameAt(event) {
  const rect = canvas.getBoundingClientRect();
  let best = null;
  let bestDistance = 11 ** 2;
  for (let index = 0; index < chartPositions.length; index += 1) {
    const [x, y] = chartPositions[index];
    const distance = (x - (event.clientX - rect.left)) ** 2 + (y - (event.clientY - rect.top)) ** 2;
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best;
}

function pinGroup(index) {
  pinnedGroup = pinnedGroup === index ? null : index;
  for (const match of element("matches").querySelectorAll("button")) {
    match.setAttribute("aria-pressed", String(Number(match.dataset.groupIndex) === pinnedGroup));
  }
  inspectGroup();
}

function resetInspection() {
  if (hoverFrame) cancelAnimationFrame(hoverFrame);
  hoverFrame = 0;
  pinnedGroup = null;
  hoveredGroup = null;
}

function renderComparison(values, excludedIndex = -1) {
  const query = unitVector(values, reference.dimension);
  const matches = closestGroups(query, vectors, reference.groups.length, excludedIndex);
  const best = reference.groups[matches[0].index];
  comparison = { matches, coordinates: project(query, reference.projection) };
  resetInspection();
  const nearest = labels[best.label].name;
  const cutoff = reference.validation.nearestDistance.p95;
  const distant = matches[0].distance > cutoff;
  element("comparison-title").textContent = distant && excludedIndex === -1 ? "Distant reference match" : nearest;
  const proximity = element("reference-proximity");
  proximity.hidden = false;
  proximity.dataset.distant = String(distant);
  proximity.textContent = distant
    ? "The closest group is farther away than for 95% of held-out project crops."
    : "This distance is within the range of 95% of held-out project crops.";
  element("comparison-note").textContent = excludedIndex !== -1
    ? "Dataset label of the closest group. The sample’s own group is excluded."
    : "Dataset label of the closest group. A close match only describes resemblance within this reference set.";
  element("map-description").textContent =
    "A PCA scatter plot of 603 reference scan groups. The displayed query is closest to a group labeled " + nearest + ".";
  const list = element("matches");
  list.replaceChildren(...matches.map(({ index, distance }) => {
    const group = reference.groups[index];
    const row = document.createElement("li");
    row.className = "match";
    const label = document.createElement("strong");
    label.textContent = labels[group.label].name;
    const groupInfo = document.createElement("span");
    groupInfo.className = "group-meta";
    const id = String(group.id).match(/\d+$/)?.[0];
    groupInfo.textContent = (id ? "Group " + id + " · " : "") + group.sourceCount + " crops";
    const value = document.createElement("span");
    value.className = "match-value";
    value.textContent = distance.toFixed(3);
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.append(swatch(group.label), label, groupInfo, value);
    button.dataset.groupIndex = String(index);
    button.addEventListener("click", () => pinGroup(index));
    row.append(button);
    return row;
  }));
  inspectGroup();
}

function resetComparison(title, note) {
  comparison = null;
  resetInspection();
  element("reference-proximity").hidden = true;
  element("comparison-title").textContent = title;
  element("comparison-note").textContent = note;
  element("matches").replaceChildren();
  element("map-caption").textContent = "The map compresses 2,048 features into two dimensions. Neighbor ranking uses the full feature vectors.";
  element("map-description").textContent = "A scatter plot of reference scan groups in the first two principal components, colored by dataset label.";
  inspectGroup();
}

function useSample(sample, animate = false) {
  latestRun += 1;
  if (pendingUpload) {
    worker?.terminate();
    worker = undefined;
    pendingUpload = null;
  }
  pick.disabled = false;
  markSelection(sample.id);
  element("atlas-link-text").textContent = "See the reference atlas";
  showImage(localAsset(sample.imagePath).href, "Published MRI project crop labeled " + labels[sample.label].name,
    "Published example", "This crop uses its saved ResNet50 features.");
  if (animate && !reducedMotion.matches && typeof image.animate === "function") {
    for (const animation of image.getAnimations()) animation.cancel();
    image.animate([{ opacity: 0.52, transform: "translateY(4px)" }, { opacity: 1, transform: "translateY(0)" }],
      { duration: 170, easing: "cubic-bezier(.23,1,.32,1)" });
  }
  renderComparison(sample.embedding, sample.groupIndex);
  say("Showing the published " + labels[sample.label].name.toLowerCase() + " example.");
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
        element("atlas-link-text").textContent = "See your crop in the atlas";
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
      "The image stays in this tab. Use a de-identified MRI crop similar to the project’s source images.", true);
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
      const thumb = document.createElement("img");
      thumb.alt = "";
      thumb.src = localAsset(sample.imagePath).href;
      const caption = document.createElement("span");
      caption.append(swatch(sample.label), labels[sample.label].name);
      button.append(thumb, caption);
      button.addEventListener("click", (event) => useSample(sample, event.detail > 0));
      samples.append(button);
    }
    samples.setAttribute("aria-busy", "false");
    const legend = element("legend");
    for (const [key, label] of Object.entries(labels)) {
      const entry = document.createElement("span");
      entry.append(swatch(key), label.name);
      legend.append(entry);
    }
    const queryLegend = document.createElement("span");
    queryLegend.append(swatch(null, "query-key"), "Query");
    legend.append(queryLegend);
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
    samples.setAttribute("aria-busy", "false");
    say("The comparison files could not load. Refresh to retry. The original findings are below.", true);
    console.error(error);
  }
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
canvas.addEventListener("pointermove", (event) => {
  if (event.pointerType !== "mouse" || hoverFrame) return;
  hoverFrame = requestAnimationFrame(() => {
    hoverFrame = 0;
    const next = frameAt(event);
    if (hoveredGroup !== next) {
      hoveredGroup = next;
      inspectGroup();
    }
  });
});
canvas.addEventListener("click", (event) => {
  const index = frameAt(event);
  if (index !== null) pinGroup(index);
});
canvas.addEventListener("pointerleave", () => {
  if (hoverFrame) cancelAnimationFrame(hoverFrame);
  hoverFrame = 0;
  if (hoveredGroup !== null) {
    hoveredGroup = null;
    inspectGroup();
  }
});
start();
