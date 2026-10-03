/* sysone-bench — chart rendering
 *
 * Every chart is built as inline SVG from data/results.json. No chart library: the visual language
 * has to stay consistent with the CSS tokens, and a leaderboard is not a generic bar chart.
 */
"use strict";

const A = (sel) => document.querySelector(sel);
const NS = "http://www.w3.org/2000/svg";
const el = (name, attrs = {}) => {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};
const pct = (x) => (x * 100).toFixed(2);
const fixed = (x, n = 4) => (x === null || x === undefined ? "—" : Number(x).toFixed(n));

let D = null;
let state = { filter: "all", sort: "accuracy", model: null, compare: "none" };

/* ------------------------------------------------------------------ load */
fetch("data/results.json")
  .then((r) => {
    if (!r.ok) throw new Error(`results.json ${r.status}`);
    return r.json();
  })
  .then((json) => {
    D = json;
    hydrate();
    drawAll();
    wire();
  })
  .catch((err) => {
    console.error(err);
    A("#leaderboard .band-head p").textContent =
      "Could not load data/results.json. The figures on this page are rendered from that file and are not duplicated here on purpose.";
  });

/* --------------------------------------------------------------- helpers */
const rows = () => D.measurements;

function visible() {
  let list = rows().slice();
  if (state.filter === "technique") list = list.filter((r) => r.techniqueReimplementation);
  else if (state.filter === "vendor") list = list.filter((r) => !r.techniqueReimplementation);
  if (state.sort === "accuracy") list.sort((a, b) => b.accuracy - a.accuracy);
  else if (state.sort === "runner") list.sort((a, b) => a.runner.localeCompare(b.runner));
  else {
    // Spread = mean absolute per-suite deviation. A wide spread is not a penalty, it is a shape.
    const spread = (r) => {
      const v = Object.values(r.suites);
      if (!v.length) return 0;
      const mean = v.reduce((s, x) => s + x, 0) / v.length;
      return v.reduce((s, x) => s + Math.abs(x - mean), 0) / v.length;
    };
    list.sort((a, b) => spread(a) - spread(b));
  }
  return list;
}

function kindTag(r) {
  if (r.runner === "pngwn") return '<span class="tag tag-unresolved">prompt unresolved</span>';
  if (r.techniqueReimplementation) return '<span class="tag tag-technique">technique reimpl.</span>';
  if (r.sharding) return '<span class="tag tag-shard">2-gpu shard</span>';
  return '<span class="tag">vendor readout</span>';
}

function colour(r) {
  if (r.techniqueReimplementation) return "var(--caveat)";
  if (r.runner === "pngwn") return "var(--warn)";
  return "var(--accent)";
}

/* ------------------------------------------------------------ leaderboard */
function drawLeaderboard() {
  const svg = A("#chart-leaderboard");
  svg.replaceChildren();
  const data = visible();
  const ref = D.referenceClosedApi;

  const rowH = 17;
  const padL = 168, padR = 64, padT = 30, padB = 26;
  const w = 1000;
  const h = padT + data.length * rowH + padB;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("height", h);

  const min = 0.30, max = 0.92;
  const x = (v) => padL + ((v - min) / (max - min)) * (w - padL - padR);

  for (let t = 0.3; t <= 0.9; t += 0.1) {
    svg.append(el("line", { x1: x(t), x2: x(t), y1: padT - 8, y2: h - padB, class: "grid-line" }));
    const lab = el("text", { x: x(t), y: h - padB + 14, class: "axis-label", "text-anchor": "middle" });
    lab.textContent = t.toFixed(1);
    svg.append(lab);
  }

  // The closed-API reference, drawn as a vertical rule behind the bars so the gap is visible.
  const rx = x(ref.accuracy);
  svg.append(el("line", {
    x1: rx, x2: rx, y1: padT - 14, y2: h - padB,
    stroke: "var(--closed)", "stroke-width": 1.5, "stroke-dasharray": "3 3", opacity: 0.85,
  }));
  const rl = el("text", { x: rx + 5, y: padT - 18, class: "axis-label", fill: "var(--closed)" });
  rl.textContent = `closed API ${pct(ref.accuracy)}`;
  svg.append(rl);

  data.forEach((r, i) => {
    const y = padT + i * rowH;
    const bw = Math.max(1, x(r.accuracy) - padL);
    const g = el("g", { class: "bar", tabindex: "0" });
    g.append(el("title")).textContent =
      `${r.runner} — ${pct(r.accuracy)}%  ·  ${r.model || ""}  ·  ${r.scoring || "vendor readout"}`;

    g.append(el("rect", { x: 0, y, width: w, height: rowH, fill: "transparent" }));

    const name = el("text", { x: 8, y: y + 12, class: "bar-label" });
    name.textContent = r.runner.length > 21 ? r.runner.slice(0, 20) + "…" : r.runner;
    if (r.techniqueReimplementation) name.setAttribute("fill", "var(--caveat)");
    g.append(name);

    g.append(el("rect", { x: padL, y: y + 3, width: bw, height: rowH - 7, rx: 2, fill: colour(r) }));
    const val = el("text", { x: padL + bw + 7, y: y + 12, class: "bar-value" });
    val.textContent = r.accuracy.toFixed(4);
    g.append(val);
    svg.append(g);
  });
}

/* ----------------------------------------------------------------- table */
function drawTable() {
  const body = A("#table-all tbody");
  body.replaceChildren();
  visible().forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="rank">${i + 1}</td>
      <td><code>${r.runner}</code></td>
      <td class="num">${r.accuracy.toFixed(4)}</td>
      <td>${kindTag(r)}</td>
      <td style="color:var(--ink-3);font-size:0.8rem">${r.scoring || "—"}</td>
      <td class="num" style="color:var(--ink-4);font-size:0.75rem">${r.runId}</td>`;
    body.append(tr);
  });
}

/* ---------------------------------------------------------------- suites */
function drawSuites() {
  const svg = A("#chart-suites");
  svg.replaceChildren();
  const legend = A("#suite-legend");
  legend.replaceChildren();

  const pick = state.compare === "top" ? rows().slice(0, 5) : [rows().find((r) => r.runner === state.model)];
  const series = pick.filter(Boolean);
  const suites = D.suites;

  const padL = 46, padR = 16, padT = 18, padB = 58;
  const w = 1000;
  const h = 300;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const plotH = h - padT - padB;
  const band = (w - padL - padR) / suites.length;
  const y = (v) => padT + plotH - v * plotH;

  for (let t = 0; t <= 1.0001; t += 0.25) {
    svg.append(el("line", { x1: padL, x2: w - padR, y1: y(t), y2: y(t), class: "grid-line" }));
    const lab = el("text", { x: padL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
    lab.textContent = t.toFixed(2);
    svg.append(lab);
  }

  const hues = ["var(--accent)", "var(--closed)", "var(--caveat)", "var(--ok)", "var(--warn)"];
  suites.forEach((s, si) => {
    const bx = padL + si * band;
    series.forEach((r, ri) => {
      const v = r.suites[s];
      if (v === undefined) return;
      const bw = (band * 0.72) / series.length;
      const x0 = bx + band * 0.14 + ri * bw;
      const rect = el("rect", {
        x: x0, y: y(v), width: Math.max(1, bw - 2), height: Math.max(1, plotH - (y(v) - padT)),
        rx: 2, fill: hues[ri % hues.length], class: "bar",
      });
      rect.append(el("title")).textContent = `${r.runner} · ${s} · ${pct(v)}%`;
      svg.append(rect);
    });
    const tick = el("text", {
      x: bx + band / 2, y: h - padB + 16, class: "axis-label", "text-anchor": "end",
      transform: `rotate(-42 ${bx + band / 2} ${h - padB + 16})`,
    });
    tick.textContent = s;
    svg.append(tick);
  });

  A("#suite-title").textContent =
    state.compare === "top" ? "Top 5 models, per suite" : `${state.model}, per suite`;

  series.forEach((r, i) => {
    const d = document.createElement("div");
    d.innerHTML = `<i style="background:${hues[i % hues.length]}"></i> ${r.runner}`;
    legend.append(d);
  });
}

/* --------------------------------------------------------------- heatmap */
function drawHeatmap() {
  const svg = A("#chart-heatmap");
  svg.replaceChildren();
  const list = rows().slice().sort((a, b) => b.accuracy - a.accuracy);
  const suites = D.suites;

  const padL = 150, padT = 26, cell = 15, gap = 2;
  const w = padL + suites.length * (cell + gap) + 12;
  const h = padT + list.length * (cell + gap) + 8;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("height", h);

  suites.forEach((s, si) => {
    const t = el("text", {
      x: padL + si * (cell + gap) + cell / 2, y: padT - 9,
      class: "axis-label", "text-anchor": "start",
      transform: `rotate(-52 ${padL + si * (cell + gap) + cell / 2} ${padT - 9})`,
    });
    t.textContent = s;
    svg.append(t);
  });

  list.forEach((r, ri) => {
    const y = padT + ri * (cell + gap);
    const lab = el("text", { x: 0, y: y + cell - 3, class: "bar-label" });
    lab.textContent = r.runner.length > 21 ? r.runner.slice(0, 20) + "…" : r.runner;
    svg.append(lab);

    suites.forEach((s, si) => {
      const v = r.suites[s];
      const rect = el("rect", {
        x: padL + si * (cell + gap), y, width: cell, height: cell, rx: 2,
        fill: v === undefined ? "var(--surface-2)" : `color-mix(in oklab, var(--accent) ${Math.round(v * 100)}%, var(--surface-2))`,
        opacity: r.techniqueReimplementation ? 0.55 : 1,
      });
      rect.append(el("title")).textContent = `${r.runner} · ${s} · ${v === undefined ? "no data" : pct(v) + "%"}`;
      svg.append(rect);
    });
  });
}

/* ------------------------------------------------------------- precision */
function drawPrecision() {
  const svg = A("#chart-precision");
  svg.replaceChildren();
  const host = D.measurements.find((r) => r.runner === "tev1-08b");
  const pairs = [
    ["CPU · fp32", 0.7629, "var(--closed)"],
    ["T4 · fp16", host ? host.accuracy : 0.7734, "var(--accent)"],
  ];
  const padL = 54, padT = 16, padB = 42, padR = 74;
  const w = 520, h = 220;
  const plotH = h - padT - padB;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  // Truncated axis, stated on the chart, because the whole point is a 0.0105 gap.
  const min = 0.70, max = 0.80;
  const y = (v) => padT + plotH - ((v - min) / (max - min)) * plotH;

  for (let t = 0.70; t <= 0.8001; t += 0.025) {
    svg.append(el("line", { x1: padL, x2: w - padR, y1: y(t), y2: y(t), class: "grid-line" }));
    const lab = el("text", { x: padL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
    lab.textContent = t.toFixed(3);
    svg.append(lab);
  }
  const axisNote = el("text", { x: w - padR, y: h - 8, class: "axis-label", "text-anchor": "end" });
  axisNote.textContent = "axis truncated to 0.70–0.80";
  svg.append(axisNote);

  pairs.forEach(([label, v, fill], i) => {
    const bw = 96, x0 = padL + 24 + i * (bw + 52);
    svg.append(el("rect", { x: x0, y: y(v), width: bw, height: Math.max(1, h - padB - y(v)), rx: 3, fill }));
    svg.lastChild.append(el("title")).textContent = `${label} · ${pct(v)}%`;
    const vt = el("text", { x: x0 + bw / 2, y: y(v) - 7, class: "bar-value", "text-anchor": "middle" });
    vt.textContent = v.toFixed(4);
    svg.append(vt);
    const lt = el("text", { x: x0 + bw / 2, y: h - padB + 17, class: "axis-label", "text-anchor": "middle" });
    lt.textContent = label;
    svg.append(lt);
  });

  const gapY = y(0.7629) - 26;
  const gl = el("text", { x: w - padR, y: gapY + 8, class: "axis-label", fill: "var(--warn)", "text-anchor": "end" });
  gl.textContent = "Δ 0.0105";
  svg.append(gl);
}

/* -------------------------------------------------------------- coverage */
function drawCoverage() {
  const svg = A("#chart-coverage");
  svg.replaceChildren();
  const items = [
    ["Run directories verified", D.runDirectoriesVerified, "var(--accent)"],
    ["Distinct runners", 44, "var(--accent)"],
    ["Distinct measurements", D.distinctMeasurements, "var(--ok)"],
  ];
  const padL = 176, padT = 14, rowH = 40;
  const w = 560, h = padT + items.length * rowH + 10;
  const max = Math.max(...items.map((i) => i[1]));
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("height", h);

  items.forEach(([label, v, fill], i) => {
    const y = padT + i * rowH;
    const bw = (v / max) * (w - padL - 66);
    const t = el("text", { x: 0, y: y + 21, class: "bar-label" });
    t.textContent = label;
    svg.append(t);
    svg.append(el("rect", { x: padL, y: y + 5, width: bw, height: 20, rx: 3, fill }));
    const vt = el("text", { x: padL + bw + 8, y: y + 21, class: "bar-value" });
    vt.textContent = v;
    svg.append(vt);
  });
}

/* -------------------------------------------------------------- hydrate */
const EXCLUDED = [
  ["akash-gemma", "No weights published", "The author account publishes zero models on Hugging Face. Code exists; the adapter does not."],
  ["semif", "Repository deleted", "Both known repository URLs return 404 and the author account publishes zero models."],
  ["decision-lux-9b", "Forced fp32 encoders", "Needs about 36 GiB of host RAM to stage against 31 GiB available. More VRAM does not help."],
  ["kev-9b", "Single-device server only", "The vendored serving path loads the model itself, so cross-GPU sharding never engages."],
  ["winnow-e4b", "GGUF only", "A vision-language checkpoint published solely as GGUF. Needs a different runtime."],
  ["clm-v0.1-8b", "Contrastive reranker", "Scores state-answer pairs rather than producing typed decisions."],
  ["jeff", "Loader detail unresolved", "Weights are public and small. The adapter is written; a package loader path needs fixing."],
];

function hydrate() {
  const best = rows()[0];
  A("#stat-best-name").textContent = best.runner;

  const sel = A("#suite-model");
  rows().forEach((r) => {
    const o = document.createElement("option");
    o.value = r.runner;
    o.textContent = `${r.runner} — ${r.accuracy.toFixed(4)}`;
    sel.append(o);
  });
  state.model = best.runner;

  const list = A("#collapse-list");
  const dup = D.duplicates;
  const row = (name, why) =>
    `<div style="display:flex;gap:.6rem;align-items:baseline;padding:.5rem 0;border-bottom:1px solid var(--line-soft)">
       <code style="font-size:.8rem;color:var(--ink)">${name}</code>
       <span style="font-size:.78rem;color:var(--ink-3);margin-left:auto;text-align:right">${why}</span>
     </div>`;
  for (const [runner, dirs] of Object.entries(dup.repeatedRuns || {})) {
    list.innerHTML += row(runner, `${dirs.length} directories, one model`);
  }
  for (const [fp, runners] of Object.entries(dup.identicalPredictions || {})) {
    list.innerHTML += row(runners.join(" + "), "identical predictions on all 1240 decisions");
  }

  const grid = A("#excluded-grid");
  grid.innerHTML = EXCLUDED.map(([name, reason, detail]) => `
    <div class="card">
      <h3 style="display:flex;justify-content:space-between;gap:.6rem;align-items:baseline">
        <code style="font-size:.9rem">${name}</code>
      </h3>
      <p style="color:var(--warn);font-size:.78rem;font-family:var(--font-num);margin-bottom:.5rem">${reason}</p>
      <p style="font-size:.85rem">${detail}</p>
    </div>`).join("");

  A("#foot-count").textContent =
    `${D.distinctMeasurements} measured, ${D.scopeTotal - D.distinctMeasurements} not measured.`;

  const tied = rows().slice(0, 3);
  const spread = tied[0].accuracy - tied[2].accuracy;
  A("#tie-note").textContent =
    `${tied.map((r) => r.runner).join(", ")} sit within ${spread.toFixed(4)} of each other`;
}

function drawAll() {
  drawLeaderboard();
  drawTable();
  drawSuites();
  drawHeatmap();
  drawPrecision();
  drawCoverage();
}

function wire() {
  document.querySelectorAll("[data-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.filter = btn.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      drawLeaderboard();
      drawTable();
    });
  });

  A("#sort-by").addEventListener("change", (e) => {
    state.sort = e.target.value;
    drawLeaderboard();
    drawTable();
  });

  A("#suite-model").addEventListener("change", (e) => {
    state.model = e.target.value;
    state.compare = "none";
    document.querySelectorAll("[data-compare]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.compare === "none")));
    drawSuites();
  });

  document.querySelectorAll("[data-compare]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.compare = btn.dataset.compare;
      document.querySelectorAll("[data-compare]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      drawSuites();
    });
  });

  if (!("IntersectionObserver" in window)) {
    document.querySelectorAll(".reveal").forEach((n) => n.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }),
    { rootMargin: "0px 0px -8% 0px" }
  );
  document.querySelectorAll(".reveal").forEach((n) => io.observe(n));
}
