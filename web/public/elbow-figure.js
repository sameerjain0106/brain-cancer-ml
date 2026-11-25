const figure = document.getElementById("elbow-chart");
const fallback = document.getElementById("elbow-static");
const scoresUrl = new URL("./research/clustering_metrics.json", import.meta.url);
const scoreFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 });

function percentDrop(previous, current) {
  return ((previous - current) / previous * 100).toFixed(1) + "%";
}

function renderElbow(scores) {
  figure.innerHTML = `
    <figcaption class="elbow-caption">
      <strong>Choosing the number of clusters</strong>
      <span id="elbow-takeaway"></span>
    </figcaption>
    <div class="elbow-axis-title">Within-cluster sum of squares</div>
    <canvas aria-hidden="true"></canvas>
    <div class="elbow-control">
      <label for="elbow-count">Number of clusters</label>
      <input id="elbow-count" type="range" min="1" step="1" value="3" aria-describedby="elbow-takeaway">
      <span id="elbow-selection" aria-hidden="true"></span>
    </div>
    <output class="elbow-reading" for="elbow-count" aria-live="polite"></output>`;

  const canvas = figure.querySelector("canvas");
  const context = canvas.getContext("2d");
  if (!context) return;

  const slider = figure.querySelector("input");
  const mapSlider = document.getElementById("map-cluster-count");
  const selection = figure.querySelector("#elbow-selection");
  const reading = figure.querySelector("output");
  const count = scores.length;
  const step = 5000;
  const maximum = Math.ceil(Math.max(...scores) / step) * step;
  const left = 44;
  const right = 15;

  slider.max = String(count);
  if (mapSlider) slider.value = mapSlider.value;
  figure.querySelector("#elbow-takeaway").textContent = "In the saved runs, the score fell " + percentDrop(scores[1], scores[2]) +
    " from two to three clusters and " + percentDrop(scores[2], scores[3]) + " from three to four.";

  function draw() {
    const { width, height } = canvas.getBoundingClientRect();
    const span = width - left - right;
    if (span <= 0 || height <= 36) return;

    const pixelRatio = Math.min(devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * pixelRatio);
    const pixelHeight = Math.round(height * pixelRatio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    context.setTransform(pixelWidth / width, 0, 0, pixelHeight / height, 0, 0);
    context.clearRect(0, 0, width, height);

    const top = 9;
    const base = height - 27;
    const x = index => left + index * span / (count - 1);
    const y = value => base - (base - top) * value / maximum;
    const selected = slider.valueAsNumber - 1;

    context.font = '12px "Segoe UI", system-ui, sans-serif';
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillStyle = "#39555a";
    context.strokeStyle = "#b6d2cd";
    context.lineWidth = 1;
    for (let value = 0; value <= maximum; value += step) {
      const heightAtValue = y(value);
      context.fillText(value.toLocaleString(), left - 7, heightAtValue);
      context.beginPath();
      context.moveTo(left, heightAtValue);
      context.lineTo(x(count - 1), heightAtValue);
      context.stroke();
    }

    const ticks = span >= 330 ? scores.map((_, index) => index) : [0, 2, 5, 9, count - 1];
    context.textAlign = "center";
    context.textBaseline = "top";
    for (const index of ticks) context.fillText(String(index + 1), x(index), base + 8);

    context.beginPath();
    context.moveTo(x(selected), top);
    context.lineTo(x(selected), base);
    context.strokeStyle = "#b55d38";
    context.setLineDash([4, 4]);
    context.stroke();
    context.setLineDash([]);

    context.beginPath();
    context.moveTo(x(0), y(scores[0]));
    for (let index = 1; index < count; index++) context.lineTo(x(index), y(scores[index]));
    context.strokeStyle = "#176561";
    context.lineWidth = 2.4;
    context.lineJoin = "round";
    context.stroke();

    for (let index = 0; index < count; index++) {
      context.beginPath();
      context.arc(x(index), y(scores[index]), index === selected ? 6 : 3.5, 0, Math.PI * 2);
      context.fillStyle = index === selected ? "#a34629" : "#176561";
      context.fill();
    }
  }

  function update() {
    const k = slider.valueAsNumber;
    const score = scores[k - 1];
    const clusterLabel = k === 1 ? "1 cluster" : k + " clusters";
    slider.setAttribute("aria-valuetext", clusterLabel);
    selection.textContent = clusterLabel;
    const change = k === 1 ? "Baseline run" : percentDrop(scores[k - 2], score) + " lower than with " +
      (k === 2 ? "one cluster" : (k - 1) + " clusters");
    reading.textContent = scoreFormat.format(score) + " within-cluster sum of squares. " + change + ".";
    draw();
  }

  function selectCount() {
    if (!mapSlider) { update(); return; }
    mapSlider.value = slider.value;
    mapSlider.dispatchEvent(new Event("input"));
  }

  slider.addEventListener("input", selectCount);
  mapSlider?.addEventListener("input", () => {
    slider.value = mapSlider.value;
    update();
  });
  canvas.addEventListener("click", event => {
    const bounds = canvas.getBoundingClientRect();
    const offset = event.clientX - bounds.left - left;
    const span = bounds.width - left - right;
    if (span <= 0 || offset < -6 || offset > span + 6) return;
    slider.value = String(1 + Math.round(Math.min(1, Math.max(0, offset / span)) * (count - 1)));
    selectCount();
  });

  figure.hidden = false;
  update();
  if (fallback) fallback.hidden = true;
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(draw).observe(canvas);
  else window.addEventListener("resize", draw);
}

async function loadElbow() {
  const response = await fetch(scoresUrl);
  if (!response.ok) return;
  const metrics = await response.json();
  const scores = Array.from({ length: 14 }, (_, index) => metrics.elbow_wcss?.[index + 1]);
  if (scores.every(score => Number.isFinite(score) && score > 0)) renderElbow(scores);
}

if (figure) loadElbow().catch(() => {});
