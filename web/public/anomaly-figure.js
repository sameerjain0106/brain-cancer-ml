const count = new Intl.NumberFormat("en-US");
const percentage = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 2 });

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function createRow(name, outcomes) {
  const total = outcomes.reduce((sum, outcome) => sum + outcome.value, 0);
  const row = node("li", "anomaly-row");
  const heading = node("div", "anomaly-row-heading");
  heading.append(node("strong", null, name), node("span", null, `${count.format(total)} crops`));

  const track = node("div", "anomaly-track");
  const fill = node("div", "anomaly-fill");
  track.setAttribute("aria-hidden", "true");
  track.append(fill);

  const labels = node("div", "anomaly-outcomes");
  for (const { label, value, color } of outcomes) {
    const segment = node("span", `anomaly-${color}`);
    segment.style.flexGrow = String(value);
    fill.append(segment);

    const item = node("div", `anomaly-outcome anomaly-${color}`);
    const summary = node("span");
    summary.append(node("strong", null, count.format(value)), ` · ${percentage.format(value / total)}`);
    item.append(node("span", "anomaly-outcome-name", label), summary);
    labels.append(item);
  }

  row.append(heading, track, labels);
  return { row, fill, total };
}

async function mountFigure(figure, fallback) {
  const response = await fetch(new URL("./research/anomaly_metrics.json", import.meta.url));
  if (!response.ok) return;
  const data = await response.json();
  const matrix = data.confusion_matrix?.matrix;
  if (!Array.isArray(matrix) || matrix.length !== 2 || !matrix.every(row => (
    Array.isArray(row) && row.length === 2 && row.every(value => Number.isSafeInteger(value) && value >= 0) && row[0] + row[1] > 0
  ))) return;

  const [[gliomaMatch, gliomaMiss], [otherMiss, otherMatch]] = matrix;
  const rows = [
    createRow("Held-out glioma", [
      { label: "Called glioma-like", value: gliomaMatch, color: "expected" },
      { label: "Called an anomaly", value: gliomaMiss, color: "unexpected" },
    ]),
    createRow("Other crops", [
      { label: "Called an anomaly", value: otherMatch, color: "expected" },
      { label: "Called glioma-like", value: otherMiss, color: "unexpected" },
    ]),
  ];
  const maxTotal = Math.max(...rows.map(row => row.total));
  const title = node("figcaption", "anomaly-chart-title", "How the anomaly model labeled crops");
  title.id = "anomaly-chart-title";
  figure.setAttribute("aria-labelledby", title.id);

  const modes = node("fieldset", "anomaly-modes");
  modes.append(node("legend", "sr-only", "Bar scale"));
  for (const [value, text] of [["share", "Share of each class"], ["count", "Number of crops"]]) {
    const input = node("input", "sr-only");
    input.type = "radio";
    input.name = "anomaly-scale";
    input.id = `anomaly-scale-${value}`;
    input.value = value;
    input.checked = value === "share";
    const label = node("label", null, text);
    label.htmlFor = input.id;
    modes.append(input, label);
  }

  const list = node("ul", "anomaly-chart-rows");
  list.append(...rows.map(row => row.row));
  const scale = node("p", "anomaly-scale-note");
  scale.setAttribute("role", "status");
  const takeaway = node("p", "anomaly-takeaway",
    `The model called ${count.format(otherMiss)} of ${count.format(otherMiss + otherMatch)} non-glioma crops glioma-like.`);

  function setMode(mode) {
    const showCounts = mode === "count";
    for (const { fill, total } of rows) {
      fill.style.setProperty("--anomaly-scale", String(showCounts ? total / maxTotal : 1));
    }
    scale.textContent = showCounts
      ? `Both bars use the same scale from 0 to ${count.format(maxTotal)} crops.`
      : "Each bar shows 100% of its source group.";
  }

  modes.addEventListener("change", event => setMode(event.target.value));
  setMode("share");
  figure.replaceChildren(title, modes, list, scale, takeaway);
  figure.hidden = false;
  if (fallback) fallback.hidden = true;
}

const figure = document.getElementById("anomaly-chart");
if (figure && !document.documentElement.hasAttribute("data-portfolio-embed")) {
  mountFigure(figure, document.getElementById("anomaly-static")).catch(() => {
    // Keep the original figure visible when the published metrics cannot load.
  });
}
