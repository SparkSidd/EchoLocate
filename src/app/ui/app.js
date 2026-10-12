/**
 * EchoLocate — Experiment 6.25 Interactive Research Visualization Engine
 */

let activeView = "overview";
let roomDimensions = [10.0, 6.0];
let micPositions = [[2.0, 1.5], [8.0, 1.5], [5.0, 4.5], [5.0, 1.5]];
let currentEstimate = [4.5, 3.0];
let currentGroundTruth = [4.5, 3.2];
let trajectory = [];
let explainerStep = 4;
let isDatasetPlaying = false;
let gccDataCache = null;

window.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  initOverviewCanvas();
  fetchStatusLoop();
  setInterval(fetchStatusLoop, 500);
});

function setupNavigation() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      activeView = btn.dataset.view;
      const target = document.getElementById(`view-${activeView}`);
      if (target) target.classList.add("active");
      handleViewSwitch(activeView);
    });
  });
}

function handleViewSwitch(view) {
  if (view === "signals") fetchWaveforms();
  else if (view === "gcc_deepgcc") fetchGCCAndDeepGCC();
  else if (view === "srp_heatmap") fetchSpatialHeatmap();
  else if (view === "srp_explainer") loadExplainerStep(explainerStep);
  else if (view === "method_comp") runMethodComparison();
  else if (view === "uncertainty") fetchUncertainty();
  else if (view === "tracking") fetchTracking();
  else if (view === "timeline") fetchTimeline();
  else if (view === "overview") drawOverview();
  else if (view === "hardware") initHardwareDemo();
}

async function fetchStatusLoop() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    // Top KPIs
    document.getElementById("kpi-method").textContent = data.active_method;
    if (data.latest_estimate && data.latest_estimate.x !== null) {
      currentEstimate = [data.latest_estimate.x, data.latest_estimate.y];
      document.getElementById("kpi-pos").textContent =
        `X = ${data.latest_estimate.x.toFixed(2)} m, Y = ${data.latest_estimate.y.toFixed(2)} m`;
      const confPct = Math.round(data.latest_estimate.confidence * 100);
      document.getElementById("kpi-conf").textContent = `${confPct}%`;
    }
    if (data.latency) {
      document.getElementById("kpi-lat").textContent = `${data.latency.mean_ms.toFixed(1)} ms`;
      document.getElementById("kpi-rtf").textContent = `${data.latency.rtf.toFixed(3)}x`;
    }

    if (data.trajectory_points) {
      trajectory = data.trajectory_points;
    }

    if (activeView === "overview") {
      drawOverview();
    }
  } catch (err) {
    console.warn("Status poll error:", err);
  }
}

/* --- View 1: Overview Canvas --- */
function initOverviewCanvas() {
  drawOverview();
}

function drawOverview() {
  const canvas = document.getElementById("overviewCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const margin = 40;
  const scale = Math.min((canvas.width - 2 * margin) / roomDimensions[0], (canvas.height - 2 * margin) / roomDimensions[1]);
  const originX = (canvas.width - roomDimensions[0] * scale) / 2;
  const originY = canvas.height - (canvas.height - roomDimensions[1] * scale) / 2;

  const toX = (x) => originX + x * scale;
  const toY = (y) => originY - y * scale;

  // Grid
  ctx.strokeStyle = "#1f2937";
  ctx.lineWidth = 1;
  for (let x = 0; x <= roomDimensions[0]; x += 1) {
    ctx.beginPath(); ctx.moveTo(toX(x), toY(0)); ctx.lineTo(toX(x), toY(roomDimensions[1])); ctx.stroke();
  }
  for (let y = 0; y <= roomDimensions[1]; y += 1) {
    ctx.beginPath(); ctx.moveTo(toX(0), toY(y)); ctx.lineTo(toX(roomDimensions[0]), toY(y)); ctx.stroke();
  }

  // Boundary
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.strokeRect(toX(0), toY(roomDimensions[1]), roomDimensions[0] * scale, roomDimensions[1] * scale);

  // Microphones
  micPositions.forEach((m, idx) => {
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath(); ctx.arc(toX(m[0]), toY(m[1]), 6, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px Inter";
    ctx.fillText(`M${idx + 1}`, toX(m[0]) - 8, toY(m[1]) - 10);
  });

  // Ground truth
  if (currentGroundTruth) {
    ctx.fillStyle = "#22c55e";
    ctx.beginPath(); ctx.arc(toX(currentGroundTruth[0]), toY(currentGroundTruth[1]), 7, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "#ffffff"; ctx.stroke();
  }

  // Estimated Source
  if (currentEstimate) {
    ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
    ctx.beginPath(); ctx.arc(toX(currentEstimate[0]), toY(currentEstimate[1]), 22, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.arc(toX(currentEstimate[0]), toY(currentEstimate[1]), 7, 0, 2 * Math.PI); ctx.fill();
  }
}

/* --- View 2: Waveforms & Spectrogram --- */
async function fetchWaveforms() {
  try {
    const resW = await fetch("/api/visualization/waveform");
    const dataW = await resW.json();
    drawWaveforms(dataW);

    const resS = await fetch("/api/visualization/spectrogram");
    const dataS = await resS.json();
    drawSpectrogram(dataS);
  } catch (e) {
    console.error("Waveform fetch failed:", e);
  }
}

function drawWaveforms(data) {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const nCh = data.channels.length;
  const chH = canvas.height / nCh;

  data.channels.forEach((ch, idx) => {
    const yMid = idx * chH + chH / 2;
    ctx.strokeStyle = "#374151";
    ctx.beginPath(); ctx.moveTo(0, yMid); ctx.lineTo(canvas.width, yMid); ctx.stroke();

    ctx.strokeStyle = idx === 0 ? "#38bdf8" : idx === 1 ? "#22c55e" : idx === 2 ? "#fb923c" : "#f43f5e";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const samples = ch.samples;
    for (let i = 0; i < samples.length; i++) {
      const x = (i / samples.length) * canvas.width;
      const y = yMid - samples[i] * (chH * 0.4);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px Inter";
    ctx.fillText(`${ch.channel_name} (${ch.rms_energy_db.toFixed(1)} dB)`, 10, yMid - 10);
  });
}

function drawSpectrogram(data) {
  const canvas = document.getElementById("spectrogramCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const mat = data.spectrogram_db;
  const nFreq = mat.length;
  const nTime = mat[0].length;

  const dx = canvas.width / nTime;
  const dy = canvas.height / nFreq;

  for (let f = 0; f < nFreq; f++) {
    for (let t = 0; t < nTime; t++) {
      const v = mat[f][t];
      ctx.fillStyle = `hsl(${220 - v * 200}, 80%, ${v * 60}%)`;
      ctx.fillRect(t * dx, canvas.height - (f + 1) * dy, dx + 1, dy + 1);
    }
  }
}

/* --- View 3: GCC vs DeepGCC --- */
async function fetchGCCAndDeepGCC() {
  try {
    const resRaw = await fetch("/api/visualization/gcc", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const dataRaw = await resRaw.json();
    const resDeep = await fetch("/api/visualization/deepgcc", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const dataDeep = await resDeep.json();

    gccDataCache = { raw: dataRaw, deep: dataDeep };
    renderGCCPair();
  } catch (e) {
    console.error("GCC fetch failed:", e);
  }
}

function renderGCCPair() {
  if (!gccDataCache) return;
  const pairIdx = parseInt(document.getElementById("gcc-pair-select").value);
  const rawPair = gccDataCache.raw.pairs[pairIdx];
  const deepPair = gccDataCache.deep.pairs[pairIdx];

  const canvas = document.getElementById("gccCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const taus = rawPair.tau_axis_ms;
  const rawC = rawPair.correlation_curve;
  const deepC = deepPair.neural_gcc_curve;

  const toX = (i) => (i / taus.length) * canvas.width;
  const toY = (v) => canvas.height - (v * (canvas.height - 40) + 20);

  // Baseline
  ctx.strokeStyle = "#374151";
  ctx.beginPath(); ctx.moveTo(0, toY(0)); ctx.lineTo(canvas.width, toY(0)); ctx.stroke();

  // Raw GCC curve
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < rawC.length; i++) {
    if (i === 0) ctx.moveTo(toX(i), toY(rawC[i])); else ctx.lineTo(toX(i), toY(rawC[i]));
  }
  ctx.stroke();

  // Calibrated DeepGCC curve
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < deepC.length; i++) {
    if (i === 0) ctx.moveTo(toX(i), toY(deepC[i])); else ctx.lineTo(toX(i), toY(deepC[i]));
  }
  ctx.stroke();

  document.getElementById("val-raw-psr").textContent = rawPair.peak_to_sidelobe_ratio.toFixed(2);
}

/* --- View 4: SRP Spatial Heatmap --- */
async function fetchSpatialHeatmap() {
  try {
    const res = await fetch("/api/visualization/spatial", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const data = await res.json();
    drawSpatialHeatmap(data);
  } catch (e) {
    console.error("Heatmap fetch error:", e);
  }
}

function drawSpatialHeatmap(data) {
  const canvas = document.getElementById("heatmapCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const mat = data.heatmap_matrix;
  const ny = mat.length;
  const nx = mat[0].length;
  const dx = canvas.width / nx;
  const dy = canvas.height / ny;

  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const v = mat[y][x];
      ctx.fillStyle = `hsl(${240 - v * 240}, 85%, ${v * 50 + 10}%)`;
      ctx.fillRect(x * dx, canvas.height - (y + 1) * dy, dx + 1, dy + 1);
    }
  }
}

/* --- View 5: SRP Explainer Stepper --- */
async function loadExplainerStep(step) {
  explainerStep = step;
  document.getElementById("explainer-step-indicator").textContent = `Step ${step} / 8`;
  try {
    const res = await fetch("/api/visualization/srp_explainer_step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ step_number: step, candidate_x: 3.2, candidate_y: 2.2 }),
    });
    const data = await res.json();
    document.getElementById("explainer-title").textContent = data.step_title;
    document.getElementById("explainer-desc").textContent = data.explanation;
    drawExplainerStep(data);
  } catch (e) {
    console.error("Explainer fetch error:", e);
  }
}

function nextExplainerStep() {
  if (explainerStep < 8) loadExplainerStep(explainerStep + 1);
}

function prevExplainerStep() {
  if (explainerStep > 1) loadExplainerStep(explainerStep - 1);
}

function drawExplainerStep(data) {
  const canvas = document.getElementById("explainerCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const nPairs = data.pair_samples.length;
  const w = canvas.width / nPairs;

  data.pair_samples.forEach((ps, idx) => {
    ctx.strokeStyle = "#374151";
    ctx.strokeRect(idx * w + 5, 10, w - 10, canvas.height - 20);

    ctx.fillStyle = "#38bdf8";
    ctx.font = "10px Inter";
    ctx.fillText(ps.pair_label, idx * w + 12, 30);
    ctx.fillText(`Delay: ${ps.expected_delay_ms.toFixed(1)} ms`, idx * w + 12, 45);
    ctx.fillText(`Sampled: ${ps.sampled_correlation_power.toFixed(2)}`, idx * w + 12, 60);

    // Mini correlation curve
    const curve = ps.correlation_curve;
    ctx.strokeStyle = "#94a3b8";
    ctx.beginPath();
    for (let i = 0; i < curve.length; i += 4) {
      const x = idx * w + 10 + (i / curve.length) * (w - 20);
      const y = canvas.height - 40 - curve[i] * 60;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
}

/* --- View 6: Method Comparison --- */
async function runMethodComparison() {
  try {
    const res = await fetch("/api/visualization/methods", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: "demo_2_classical_failure_srp_recovery" }),
    });
    const data = await res.json();
    const tbody = document.querySelector("#method-comp-table tbody");
    tbody.innerHTML = "";

    data.methods.forEach((m) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="color:${m.marker_color}">${m.method_name}</strong></td>
        <td class="font-mono">(${m.pred_x.toFixed(2)}, ${m.pred_y.toFixed(2)})</td>
        <td class="font-mono ${m.error_m > 1.0 ? "text-red" : "text-green"}">${m.error_m ? m.error_m.toFixed(2) + " m" : "--"}</td>
        <td>${Math.round(m.confidence * 100)}%</td>
        <td class="font-mono">${m.latency_ms.toFixed(1)} ms</td>
        <td><span class="badge ${m.is_winner ? "badge-green" : "badge-outline"}">${m.status}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error("Method comparison failed:", e);
  }
}

/* --- View 7: Uncertainty --- */
async function fetchUncertainty() {
  try {
    const res = await fetch("/api/visualization/confidence");
    const data = await res.json();
    document.getElementById("gauge-conf-pct").textContent = `${Math.round(data.current_confidence_pct)}%`;
    document.getElementById("gauge-conf-state").textContent = data.confidence_state;

    // Draw Risk Coverage
    const canvas = document.getElementById("riskCoverageCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2;
    ctx.beginPath();
    data.risk_coverage_curve.forEach((pt, i) => {
      const x = (1.0 - pt.coverage_fraction) * (canvas.width - 40) + 20;
      const y = canvas.height - (pt.mean_risk_error_m / 0.6) * (canvas.height - 40) - 20;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  } catch (e) {
    console.error("Uncertainty fetch error:", e);
  }
}

/* --- View 8: Tracking --- */
async function fetchTracking() {
  try {
    const res = await fetch("/api/visualization/history");
    const data = await res.json();
    const canvas = document.getElementById("trackingCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const pts = data.trajectory_points;
    if (pts.length < 2) return;

    ctx.strokeStyle = "rgba(244, 63, 94, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = (p.raw_x / 10.0) * canvas.width;
      const y = canvas.height - (p.raw_y / 6.0) * canvas.height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = (p.smoothed_x / 10.0) * canvas.width;
      const y = canvas.height - (p.smoothed_y / 6.0) * canvas.height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  } catch (e) {
    console.error("Tracking fetch error:", e);
  }
}

/* --- View 9: Simulation Lab --- */
async function runSimLab() {
  const rt60 = parseFloat(document.getElementById("sim-rt60").value);
  const snr = parseFloat(document.getElementById("sim-snr").value);
  const pert = parseFloat(document.getElementById("sim-pert").value);

  try {
    const res = await fetch("/api/visualization/simulation_lab", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rt60: rt60, snr: snr, mic_perturbation_mm: pert, src_x: 4.5, src_y: 3.2 }),
    });
    const data = await res.json();
    document.getElementById("sim-diag-text").textContent = data.failure_diagnosis;
  } catch (e) {
    console.error("Simulation lab error:", e);
  }
}

/* --- View 11: Timeline --- */
async function fetchTimeline() {
  try {
    const res = await fetch("/api/visualization/experiments");
    const list = await res.json();
    const container = document.getElementById("timeline-list");
    container.innerHTML = "";

    list.forEach((m) => {
      const div = document.createElement("div");
      div.className = "timeline-item";
      div.style.borderLeftColor = m.badge_color;
      div.innerHTML = `
        <div style="display:flex; justify-content:space-between;">
          <h4 style="color:${m.badge_color}">${m.experiment_id}: ${m.title}</h4>
          <span class="badge badge-outline">${m.phase}</span>
        </div>
        <p><strong>Problem:</strong> ${m.problem}</p>
        <p><strong>Discovery:</strong> ${m.discovery}</p>
        <p><strong>Failure Mechanism:</strong> ${m.failure_mechanism}</p>
        <p><strong>Next Solution:</strong> ${m.next_solution}</p>
      `;
      container.appendChild(div);
    });
  } catch (e) {
    console.error("Timeline fetch error:", e);
  }
}

/* --- View 12: Examiner Mode --- */
async function loadExaminerPreset(scenarioId) {
  try {
    await fetch("/api/scenario/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId }),
    });
    fetchStatusLoop();
  } catch (e) {
    console.error("Preset error:", e);
  }
}

async function examinerRerun() {
  try {
    const res = await fetch("/api/examiner/rerun_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method_id: "M12_Confidence_Hybrid" }),
    });
    const data = await res.json();
    document.getElementById("exam-live-lat").textContent = `${data.latency_ms.toFixed(1)} ms`;
  } catch (e) {
    console.error("Examiner rerun error:", e);
  }
}

function changeGlobalMethod(methodId) {
  fetch("/api/method", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method_id: methodId }),
  });
}

function clearVisualMap() {
  trajectory = [];
  drawOverview();
}

function toggleFullScreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen();
  } else {
    document.exitFullscreen();
  }
}
