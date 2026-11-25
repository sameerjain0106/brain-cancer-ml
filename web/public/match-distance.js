let ruler;

function getRuler() {
  const details = document.getElementById("match-distance");
  const body = document.getElementById("match-distance-body");
  if (!details || !body) return null;
  body.innerHTML = `
    <p class="match-distance-reading">This scan <strong data-ruler-distance></strong></p>
    <div class="match-distance-scale">
      <div class="match-distance-track" aria-hidden="true">
        <span class="match-distance-tick match-distance-median"></span>
        <span class="match-distance-tick match-distance-p90"></span>
        <span class="match-distance-tick match-distance-cutoff"></span>
        <span class="match-distance-current"></span>
      </div>
      <div class="match-distance-endpoints"><span><span class="sr-only">Scale starts at </span>0</span><span>Study maximum <span data-ruler-max></span></span></div>
    </div>
    <dl class="match-distance-values">
      <div><dt>Study median</dt><dd data-ruler-median></dd></div>
      <div><dt>90th percentile</dt><dd data-ruler-p90></dd></div>
      <div><dt>Distant cutoff</dt><dd data-ruler-cutoff></dd></div>
    </dl>
    <p class="match-distance-outcome" data-ruler-outcome role="status"></p>
    <p class="match-distance-context">Smaller cosine distances mean closer matches. The cutoff is the 95th percentile for held-out crops from this study.</p>
  `;
  const read = (key) => body.querySelector(`[data-ruler-${key}]`);
  return {
    details,
    track: body.querySelector(".match-distance-track"),
    distance: read("distance"),
    median: read("median"),
    p90: read("p90"),
    cutoff: read("cutoff"),
    max: read("max"),
    outcome: read("outcome"),
  };
}

function formatDistance(value) {
  return value === 0 ? "0" : value.toFixed(5);
}

export function setMatchDistance(distance, referenceDistances) {
  const { p50, p90, p95, max } = referenceDistances ?? {};
  if (![distance, p50, p90, p95, max].every(Number.isFinite) ||
      distance < 0 || p50 < 0 || p90 < p50 || p95 < p90 || max < p95 || max <= 0) {
    clearMatchDistance();
    return;
  }
  ruler ??= getRuler();
  if (!ruler) return;

  const place = (name, value) => ruler.track.style.setProperty(name, `${Math.min(value / max, 1) * 100}%`);
  const distant = distance > p95;
  const beyondScale = distance > max;
  // Keep the scale fixed while visitors compare scans. The full distance is always shown above.
  ruler.track.dataset.beyondScale = String(beyondScale);
  place("--distance-median", p50);
  place("--distance-p90", p90);
  place("--distance-cutoff", p95);
  place("--distance-current", distance);
  ruler.distance.textContent = formatDistance(distance);
  ruler.median.textContent = formatDistance(p50);
  ruler.p90.textContent = formatDistance(p90);
  ruler.cutoff.textContent = formatDistance(p95);
  ruler.max.textContent = formatDistance(max);
  ruler.outcome.classList.toggle("is-distant", distant);
  ruler.outcome.textContent = beyondScale
    ? "This distance exceeds the study maximum. The marker is at the right edge."
    : distant
      ? "This scan is above the study’s distant cutoff."
      : "This scan is within the study’s distance cutoff.";
  ruler.details.hidden = false;
}

export function clearMatchDistance() {
  const details = ruler?.details ?? document.getElementById("match-distance");
  if (!details) return;
  details.hidden = true;
  details.open = false;
}
