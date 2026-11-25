import { labels, prepareImage, project, rankGroups, unitVector } from "./analysis.js";

const element = (id) => document.getElementById(id);
const samples = element("samples");
const status = element("status");
const image = element("query-image");
const pick = element("pick-image");
const upload = element("upload");
const canvas = element("feature-map");
const rankRows = element("rank-rows");
const distanceButtons = Array.from(document.querySelectorAll("[data-distance]"));
let distanceMode = "features";
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
let rankButtons = [];

function say(message, error = false, visible = true) {
  status.classList.toggle("sr-only", !visible);
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
  const bounds = canvas.getBoundingClientRect();
  const { width, height } = bounds;
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
  if (!reference) return;
  const padding = width > 520 ? 34 : 19;
  const groups = reference.groups;
  const xs = groups.map((group) => group.x);
  const ys = groups.map((group) => group.y);
  const extent = (values, query) => {
    const low = Math.min(...values);
    const high = Math.max(...values);
    const space = (high - low) * 0.23;
    return [
      Math.min(low, Math.max(low - space, query ?? low)),
      Math.max(high, Math.min(high + space, query ?? high)),
    ];
  };
  const [queryX, queryY] = comparison?.coordinates ?? [];
  const [minX, maxX] = extent(xs, queryX);
  const [minY, maxY] = extent(ys, queryY);
  const scale = Math.min((width - 2 * padding) / (maxX - minX), (height - 2 * padding) / (maxY - minY));
  const middleX = (minX + maxX) * 0.5;
  const middleY = (minY + maxY) * 0.5;
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const position = (x, y) => [
    width * .5 + (clamp(x, minX, maxX) - middleX) * scale,
    height * .5 - (clamp(y, minY, maxY) - middleY) * scale,
  ];

  context.strokeStyle = "#a1a897";
  context.beginPath();
  context.moveTo(padding, padding);
  context.lineTo(padding, height - padding);
  context.lineTo(width - padding, height - padding);
  for (let step = 1; step < 8; step += 1) {
    const x = padding + (width - 2 * padding) * step / 8;
    const y = padding + (height - 2 * padding) * step / 8;
    context.moveTo(x, height - padding - 3);
    context.lineTo(x, height - padding + 3);
    context.moveTo(padding - 3, y);
    context.lineTo(padding + 3, y);
  }
  context.stroke();
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
  const [qx, qy] = position(queryX, queryY);
  for (const match of comparison[distanceMode].slice(0, 3)) {
    const group = groups[match.index];
    const [gx, gy] = position(group.x, group.y);
    context.setLineDash([3, 4]);
    context.strokeStyle = distanceMode === "map" ? "#a34a3299" : "#17466cbf";
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
  context.fillStyle = "#173e61";
  context.beginPath();
  context.arc(qx, qy, 6, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.lineWidth = 1;
  const outside = queryX < minX || queryX > maxX || queryY < minY || queryY > maxY;
  element("map-caption").textContent = outside
    ? "The crop projects beyond the chart. Its marker sits at the nearest edge."
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

function inspectGroup() {
  const index = hoveredGroup ?? pinnedGroup;
  const group = index === null ? null : reference?.groups[index];
  const note = element("map-inspection");
  note.textContent = group
    ? labels[group.label].name + " group " + scanGroupName(group) + " (" + group.sourceCount + " crops)"
    : "Select a point or a match.";
  note.dataset.inspecting = String(Boolean(group));
  for (const button of rankButtons) {
    const linked = String(Number(button.dataset.groupIndex) === index);
    button.dataset.linked = linked;
    button.closest("tr").dataset.linked = linked;
  }
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
  for (const match of rankButtons) {
    match.setAttribute("aria-pressed", String(Number(match.dataset.groupIndex) === pinnedGroup));
  }
  describeRanks();
  inspectGroup();
}

function resetInspection() {
  if (hoverFrame) cancelAnimationFrame(hoverFrame);
  hoverFrame = 0;
  pinnedGroup = null;
  hoveredGroup = null;
}

function scanGroupName(group) {
  return String(group.id).match(/\d+$/)?.[0] ?? group.id;
}

function ordinal(rank) {
  const suffix = rank % 100 >= 11 && rank % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][rank % 10] ?? "th";
  return rank + suffix;
}

function describeRanks() {
  if (!comparison) return;
  const index = pinnedGroup ?? comparison.map[0].index;
  const group = reference.groups[index];
  const prefix = labels[group.label].name + " group " + scanGroupName(group);
  const featuresRank = comparison.ranks.features.get(index);
  const mapRank = comparison.ranks.map.get(index);
  element("rank-detail").textContent = featuresRank === undefined
    ? prefix + " is the source group for this example and is excluded from both rankings."
    : prefix + " ranks " + ordinal(mapRank) + " on the map and " + ordinal(featuresRank) + " across all features.";
}

function renderComparison(values, excludedIndex = -1) {
  const query = unitVector(values, reference.dimension);
  const features = rankGroups(query, vectors, reference.groups.length, excludedIndex);
  const coordinates = project(query, reference.projection);
  const map = reference.groups
    .map((group, index) => ({ index, distance: Math.hypot(coordinates[0] - group.x, coordinates[1] - group.y) }))
    .filter(({ index }) => index !== excludedIndex)
    .sort((a, b) => a.distance - b.distance || a.index - b.index);
  const rankIndex = (ranking) => new Map(ranking.map(({ index }, position) => [index, position + 1]));
  comparison = { features, map, coordinates, ranks: { features: rankIndex(features), map: rankIndex(map) } };
  const best = reference.groups[features[0].index];
  const nearest = labels[best.label].name;
  const cutoff = reference.validation.nearestDistance.p95;
  const distant = features[0].distance > cutoff;
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
  renderRankings();
}

function describeMap() {
  if (!comparison) return;
  const group = reference.groups[comparison[distanceMode][0].index];
  element("map-description").textContent =
    "A PCA scatter plot of 603 scan groups. The first group in the selected ranking is " +
    labels[group.label].name + ", group " + scanGroupName(group) + ".";
}

function rankRow({ index, distance }) {
  const group = reference.groups[index];
  const featuresRank = comparison.ranks.features.get(index);
  const mapRank = comparison.ranks.map.get(index);
  const row = document.createElement("tr");
  const heading = document.createElement("th");
  heading.scope = "row";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "rank-group";
  button.dataset.groupIndex = index;
  button.setAttribute("aria-pressed", "false");

  const groupName = document.createElement("strong");
  groupName.className = "rank-group-id";
  groupName.textContent = "Group " + scanGroupName(group);
  const groupLabel = document.createElement("span");
  groupLabel.className = "rank-group-label";
  groupLabel.append(swatch(group.label), labels[group.label].name);
  button.append(groupName, groupLabel);
  button.addEventListener("click", () => pinGroup(index));
  button.addEventListener("pointerenter", (event) => {
    if (event.pointerType === "mouse") {
      hoveredGroup = index;
      inspectGroup();
    }
  });
  button.addEventListener("pointerleave", () => {
    if (hoveredGroup === index) {
      hoveredGroup = null;
      inspectGroup();
    }
  });
  heading.append(button);
  row.append(heading);
  for (const [rank, score] of [[featuresRank, distance], [mapRank, comparison.map[mapRank - 1].distance]]) {
    const cell = row.insertCell();
    cell.className = "rank-value";
    if (rank === 1) cell.dataset.best = "true";
    const number = document.createElement("strong");
    number.className = "rank-number";
    number.textContent = rank;
    const value = document.createElement("small");
    value.className = "rank-distance";
    value.textContent = "Distance " + score.toFixed(3);
    cell.append(number, " ", value);
  }
  return row;
}

function renderRankings() {
  resetInspection();
  const candidates = new Set([...comparison.features.slice(0, 3), ...comparison.map.slice(0, 3)].map(({ index }) => index));
  const matches = comparison.features.filter(({ index }) => candidates.has(index));
  rankRows.replaceChildren(...matches.map(rankRow));
  rankButtons = Array.from(rankRows.querySelectorAll("button"));
  describeRanks();
  describeMap();
  inspectGroup();
}

function resetComparison(title, note) {
  comparison = null;
  resetInspection();
  element("reference-proximity").hidden = true;
  element("comparison-title").textContent = title;
  element("comparison-note").textContent = note;
  element("rank-detail").textContent = note;
  rankRows.replaceChildren();
  rankButtons = [];
  element("map-caption").textContent = "The two axes use the same scale.";
  element("map-description").textContent = "A scatter plot of reference scan groups in the first two principal components, colored by dataset label.";
  inspectGroup();
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

for (const button of distanceButtons) {
  button.addEventListener("click", () => {
    distanceMode = button.dataset.distance;
    for (const control of distanceButtons) {
      control.setAttribute("aria-pressed", String(control === button));
    }
    describeMap();
    drawMap();
  });
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
