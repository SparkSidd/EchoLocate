/**
 * EchoLocate — Experiment 6.5 Real Hardware Scientific Live Dashboard
 * ===================================================================
 * Mathematically rigorous visualization:
 *   1. 5-point Sinc sub-sample peak interpolation & Cramér-Rao Lower Bound (CRLB) error bars
 *   2. Full Conic Analytic Geometry: Active Hyperbolic Branch, Shadow Branch, Asymptotes & Bisector
 *   3. 1D Adaptive Kalman Filter trajectory with dynamic +/- 1-sigma uncertainty ribbon
 *   4. ISO 9613-1 Temperature-dependent acoustic speed c(T) and Far-Field AoA ray
 */

let hwPollInterval = null;
const MAX_HISTORY = 80;
const tdoaHistory = [];       // Kalman-filtered TDOA ms values
const rawTdoaHistory = [];    // Raw Sinc-interpolated TDOA ms values
const sigmaHistory = [];      // CRLB 1-sigma error in ms
const confHistory = [];       // Confidence %
const psrHistory = [];        // Peak-to-Secondary Ratio

function initHardwareDemo() {
  if (!hwPollInterval) {
    hwPollInterval = setInterval(fetchHardwareStatus, 180); // ~5.5 Hz
  }
}
initHardwareDemo();

async function fetchHardwareStatus() {
  try {
    const res = await fetch("/api/hardware/status");
    if (!res.ok) return;
    const data = await res.json();
    updateHardwareUI(data);
  } catch (err) {
    console.warn("Hardware status poll error:", err);
  }
}

// ─── Quality color mapping ───────────────────────────────────────────────────
function qualityColor(psr) {
  if (!psr || psr < 1.4) return "#ef4444";
  if (psr < 2.5)          return "#f59e0b";
  return "#22c55e";
}

function confidenceColor(conf) {
  if (!conf || conf < 30) return "#ef4444";
  if (conf < 60)          return "#f59e0b";
  return "#22c55e";
}

// ─── Main UI update ──────────────────────────────────────────────────────────
function updateHardwareUI(data) {
  if (!data) return;

  // 1. Frame hash & ID
  const hashEl = document.getElementById("hw-hash-val");
  if (hashEl && data.computation_hash) hashEl.textContent = data.computation_hash;

  // 2. Connection status
  const phoneStatus = document.getElementById("hw-phone-status");
  if (phoneStatus) {
    phoneStatus.textContent = data.is_phone_connected ? "🟢 LIVE 16kHz" : "⚫ Waiting for phone";
    phoneStatus.style.color = data.is_phone_connected ? "#22c55e" : "#9ca3af";
  }

  // 3. Laptop metrics
  if (data.laptop_metrics) {
    const m = data.laptop_metrics;
    const el = document.getElementById("hw-lap-rms");
    if (el) el.textContent = `${m.rms_energy_db.toFixed(1)} dB`;
    const freqEl = document.getElementById("hw-lap-freq");
    if (freqEl) freqEl.textContent = `${Math.round(m.dominant_freq_hz)} Hz`;

    // Animate VU bar
    const vuBar = document.getElementById("hw-vu-lap");
    if (vuBar) {
      const pct = Math.min(100, Math.max(0, (m.rms_energy_db + 60) * 1.8));
      vuBar.style.width = `${pct}%`;
      vuBar.style.background = m.rms_energy_db > -30
        ? "linear-gradient(90deg, #22c55e, #84cc16)"
        : "linear-gradient(90deg, #38bdf8, #6366f1)";
    }
  }

  // 4. Phone metrics
  if (data.phone_metrics) {
    const m = data.phone_metrics;
    const el = document.getElementById("hw-ph-rms");
    if (el) el.textContent = `${m.rms_energy_db.toFixed(1)} dB`;
    const vuBar = document.getElementById("hw-vu-phone");
    if (vuBar) {
      const pct = Math.min(100, Math.max(0, (m.rms_energy_db + 60) * 1.8));
      vuBar.style.width = `${pct}%`;
    }
  } else {
    const el = document.getElementById("hw-ph-rms");
    if (el) el.textContent = "—";
    const vuBar = document.getElementById("hw-vu-phone");
    if (vuBar) vuBar.style.width = "0%";
  }

  // 5. Clock sync & Environment
  if (data.clock_sync_state) {
    const s = data.clock_sync_state;
    const dv = document.getElementById("hw-drift-val");
    if (dv) dv.textContent = `${s.clock_drift_ppm.toFixed(1)} ppm`;
    const jv = document.getElementById("hw-jitter-val");
    if (jv) jv.textContent = `${s.jitter_ms.toFixed(2)} ms`;
    const rv = document.getElementById("hw-rtt-val");
    if (rv) rv.textContent = `${s.mean_rtt_ms.toFixed(1)} ms`;
  }

  if (data.mic_gain_offset_db !== undefined) {
    const gEl = document.getElementById("hw-gain-calib-val");
    if (gEl) gEl.textContent = `${data.mic_gain_offset_db > 0 ? "+" : ""}${data.mic_gain_offset_db.toFixed(1)} dB`;
  }

  // 6. TDOA & Mathematical Estimation
  if (data.tdoa_estimate) {
    const tdoa = data.tdoa_estimate;
    window.latestTDOA = tdoa;

    const tv = document.getElementById("hw-tdoa-val");
    if (tv) {
      tv.innerHTML = `${tdoa.tdoa_ms.toFixed(3)} <span style="font-size:10px;color:#9ca3af;">±${tdoa.sigma_tdoa_ms.toFixed(3)}ms</span>`;
      tv.style.color = qualityColor(tdoa.peak_psr);
    }
    const dv = document.getElementById("hw-dist-diff");
    if (dv) dv.textContent = `${(tdoa.distance_diff_m * 100).toFixed(1)} cm`;
    const dirv = document.getElementById("hw-dir-label");
    if (dirv) dirv.textContent = tdoa.direction_label;

    const cv = document.getElementById("hw-conf-pct");
    if (cv) {
      cv.textContent = `${Math.round(tdoa.confidence_pct)}%`;
      cv.style.color = confidenceColor(tdoa.confidence_pct);
    }

    const qv = document.getElementById("hw-sig-qual");
    if (qv && tdoa.signal_quality) {
      const sq = tdoa.signal_quality;
      qv.textContent = `${sq.signal_type} (${sq.bandwidth_score_pct.toFixed(0)}%) BW:${Math.round(sq.effective_bandwidth_hz)}Hz`;
      qv.style.color = sq.is_favorable_for_tdoa ? "#22c55e" : "#f59e0b";

      // ILD Energy Balance Indicator
      const ildEl = document.getElementById("hw-ild-val");
      if (ildEl && sq.ild_db !== undefined) {
        const ild = sq.ild_db;
        if (Math.abs(ild) < 1.5) {
          ildEl.innerHTML = `<span style="color:#22c55e;">Balanced (ΔL = ${ild > 0 ? "+" : ""}${ild.toFixed(1)} dB)</span>`;
        } else if (ild > 1.5) {
          ildEl.innerHTML = `<span style="color:#38bdf8;">⬅ Laptop is +${ild.toFixed(1)} dB louder</span>`;
        } else {
          ildEl.innerHTML = `<span style="color:#22c55e;">Phone is +${Math.abs(ild).toFixed(1)} dB louder ➡</span>`;
        }
      }
    }

    const psrv = document.getElementById("hw-psr-val");
    if (psrv) {
      psrv.textContent = `PSR ${tdoa.peak_psr.toFixed(2)} | c=${tdoa.speed_of_sound_mps.toFixed(1)}m/s (${tdoa.temperature_c}°C)`;
      psrv.style.color = qualityColor(tdoa.peak_psr);
    }

    const aoav = document.getElementById("hw-aoa-val");
    if (aoav) aoav.textContent = `${tdoa.angle_of_arrival_deg.toFixed(1)}°`;

    // Push history
    tdoaHistory.push(tdoa.tdoa_ms);
    rawTdoaHistory.push(tdoa.raw_tdoa_ms || tdoa.tdoa_ms);
    sigmaHistory.push(tdoa.sigma_tdoa_ms || 0.1);
    confHistory.push(tdoa.confidence_pct);
    psrHistory.push(tdoa.peak_psr);
    if (tdoaHistory.length > MAX_HISTORY) {
      tdoaHistory.shift();
      rawTdoaHistory.shift();
      sigmaHistory.shift();
      confHistory.shift();
      psrHistory.shift();
    }
  }

  // 7. GCC canvas
  if (data.gcc_result) drawLiveGCCCanvas(data.gcc_result, data.tdoa_estimate);

  // 8. Hyperbola canvas
  if (data.hyperbolic_locus) drawHyperbolicLocusCanvas(data.hyperbolic_locus, data.tdoa_estimate);

  // 9. TDOA timeline
  drawTDOATimeline();
}

// ─── GCC-PHAT Canvas (Mathematical Annotations) ──────────────────────────────
function drawLiveGCCCanvas(gcc, tdoa) {
  const canvas = document.getElementById("hwGccCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const curve = gcc.correlation_curve;
  const taus = gcc.tau_axis_ms;
  if (!curve || curve.length < 2) return;

  const minVal = Math.min(...curve);
  const maxVal = Math.max(...curve);
  const range = maxVal - minVal || 1e-6;
  const pad = 0.12;

  const toX = (i) => (i / (curve.length - 1)) * W;
  const toY = (v) => H - ((v - minVal) / range) * H * (1 - pad * 2) - H * pad;

  // Background grid
  ctx.strokeStyle = "#1f2937";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = (g / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Zero-delay center reference (tau = 0)
  const zeroIdx = taus.findIndex(t => t >= 0);
  if (zeroIdx >= 0) {
    ctx.strokeStyle = "rgba(156, 163, 175, 0.4)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(toX(zeroIdx), 0); ctx.lineTo(toX(zeroIdx), H); ctx.stroke();
    ctx.setLineDash([]);
  }

  const psr = gcc.psr || 1.0;
  const curveColor = qualityColor(psr);

  // GCC curve fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, curveColor === "#22c55e" ? "rgba(34,197,94,0.25)" : (curveColor === "#f59e0b" ? "rgba(245,158,11,0.25)" : "rgba(239,68,68,0.25)"));
  grad.addColorStop(1, "rgba(17,24,39,0.0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(toX(0), H);
  curve.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
  ctx.lineTo(toX(curve.length - 1), H);
  ctx.closePath();
  ctx.fill();

  // GCC curve stroke
  ctx.strokeStyle = curveColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  curve.forEach((v, i) => {
    if (i === 0) ctx.moveTo(toX(i), toY(v)); else ctx.lineTo(toX(i), toY(v));
  });
  ctx.stroke();

  // Peak marker & CRLB 1-sigma uncertainty bracket
  let maxIdx = 0;
  curve.forEach((v, i) => { if (v > curve[maxIdx]) maxIdx = i; });
  const px = toX(maxIdx), py = toY(curve[maxIdx]);

  if (tdoa && tdoa.sigma_tdoa_ms) {
    const dt_ms_per_pixel = (taus[taus.length - 1] - taus[0]) / W;
    const sigma_px = Math.max(4, tdoa.sigma_tdoa_ms / (dt_ms_per_pixel || 0.05));
    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px - sigma_px, py);
    ctx.lineTo(px + sigma_px, py);
    ctx.moveTo(px - sigma_px, py - 4); ctx.lineTo(px - sigma_px, py + 4);
    ctx.moveTo(px + sigma_px, py - 4); ctx.lineTo(px + sigma_px, py + 4);
    ctx.stroke();
  }

  // Peak dot
  ctx.fillStyle = curveColor;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

  // Header mathematical annotation
  ctx.fillStyle = "#f9fafb";
  ctx.font = "bold 10px Inter, monospace";
  const tauVal = tdoa ? tdoa.tdoa_ms.toFixed(3) : (taus[maxIdx] ? taus[maxIdx].toFixed(3) : "0.000");
  const sigmaVal = tdoa ? ` ±${tdoa.sigma_tdoa_ms.toFixed(3)}ms` : "";
  ctx.fillText(`τ_opt = ${tauVal}${sigmaVal} | PSR=${psr.toFixed(2)}`, 8, 14);

  // Math formula tag
  ctx.fillStyle = "#9ca3af";
  ctx.font = "9px Inter, sans-serif";
  ctx.fillText(`R_12(τ) = IFFT[ S_1 S_2* / |S_1 S_2*| ]`, W - 180, 14);
}

// ─── Hyperbolic Locus Canvas (Analytic Conic Geometry) ────────────────────────
function drawHyperbolicLocusCanvas(locus, tdoa) {
  const canvas = document.getElementById("hwHyperbolaCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const roomW = 10.0, roomH = 6.0;
  const scaleX = W / roomW, scaleY = H / roomH;
  const toX = x => x * scaleX;
  const toY = y => H - y * scaleY;  // flip Y (room coords: 0=bottom)

  // Room boundary
  ctx.strokeStyle = "#374151";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(0, 0, W, H);

  // 1-meter room grid
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 1;
  for (let gx = 1; gx < roomW; gx++) {
    ctx.beginPath(); ctx.moveTo(toX(gx), 0); ctx.lineTo(toX(gx), H); ctx.stroke();
  }
  for (let gy = 1; gy < roomH; gy++) {
    ctx.beginPath(); ctx.moveTo(0, toY(gy)); ctx.lineTo(W, toY(gy)); ctx.stroke();
  }

  const m1 = locus.mic1_xy || [2.0, 2.0];
  const m2 = locus.mic2_xy || [4.0, 2.0];

  // 1. Perpendicular Bisector (Reference line Delta t = 0)
  if (locus.bisector_line && locus.bisector_line.length === 2) {
    const [b0, b1] = locus.bisector_line;
    ctx.strokeStyle = "rgba(107, 114, 128, 0.4)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(toX(b0[0]), toY(b0[1])); ctx.lineTo(toX(b1[0]), toY(b1[1])); ctx.stroke();
    ctx.setLineDash([]);
  }

  // 2. Asymptotes (y = +/- (b/a)x)
  if (locus.asymptote_lines && locus.asymptote_lines.length > 0) {
    ctx.strokeStyle = "rgba(147, 51, 234, 0.25)";
    ctx.setLineDash([2, 4]);
    ctx.lineWidth = 1;
    locus.asymptote_lines.forEach(line => {
      if (line.length === 2) {
        ctx.beginPath(); ctx.moveTo(toX(line[0][0]), toY(line[0][1])); ctx.lineTo(toX(line[1][0]), toY(line[1][1])); ctx.stroke();
      }
    });
    ctx.setLineDash([]);
  }

  // 3. Baseline segment connecting M1 and M2
  ctx.strokeStyle = "rgba(156, 163, 175, 0.35)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(toX(m1[0]), toY(m1[1])); ctx.lineTo(toX(m2[0]), toY(m2[1])); ctx.stroke();

  // 4. Shadow Conjugate Hyperbola Branch (Dashed)
  if (locus.shadow_branch_points && locus.shadow_branch_points.length > 1) {
    ctx.strokeStyle = "rgba(244, 63, 94, 0.25)";
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    locus.shadow_branch_points.forEach((p, i) => {
      if (i === 0) ctx.moveTo(toX(p[0]), toY(p[1])); else ctx.lineTo(toX(p[0]), toY(p[1]));
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 5. Active Physical Hyperbolic Locus (Thick glowing curve)
  if (locus.locus_points && locus.locus_points.length > 1) {
    const tdoaVal = locus.tdoa_ms || 0;
    const locusColor = Math.abs(tdoaVal) < 0.2 ? "#a78bfa" : (Math.abs(tdoaVal) < 2.0 ? "#f59e0b" : "#38bdf8");
    ctx.strokeStyle = locusColor;
    ctx.lineWidth = 3;
    ctx.shadowColor = locusColor;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    locus.locus_points.forEach((p, i) => {
      if (i === 0) ctx.moveTo(toX(p[0]), toY(p[1])); else ctx.lineTo(toX(p[0]), toY(p[1]));
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // 6. Microphones M1 (Laptop) & M2 (Phone)
  // Laptop Mic M1
  ctx.fillStyle = "#38bdf8";
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(toX(m1[0]), toY(m1[1]), 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#e0f2fe";
  ctx.font = "bold 10px Inter, sans-serif";
  ctx.fillText("💻 M1 (Laptop)", toX(m1[0]) - 24, toY(m1[1]) - 12);

  // Phone Mic M2
  ctx.fillStyle = "#22c55e";
  ctx.strokeStyle = "#16a34a";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(toX(m2[0]), toY(m2[1]), 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#dcfce7";
  ctx.fillText("📱 M2 (Phone)", toX(m2[0]) - 20, toY(m2[1]) - 12);

  // 7. Estimated Speaker Position & Angle of Arrival Ray
  if (tdoa && tdoa.estimated_xy && tdoa.confidence_pct > 15) {
    const spk = tdoa.estimated_xy;
    const conf = tdoa.confidence_pct;
    const midX = (m1[0] + m2[0]) / 2;
    const midY = (m1[1] + m2[1]) / 2;

    // Far-field bearing ray from midpoint to estimated point
    ctx.strokeStyle = "rgba(34, 197, 94, 0.45)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(toX(midX), toY(midY));
    ctx.lineTo(toX(spk[0]), toY(spk[1]));
    ctx.stroke();
    ctx.setLineDash([]);

    // Cramér-Rao Confidence Error Ellipse
    const errRadiusX = Math.max(10, (100 - conf) * 0.4);
    const errRadiusY = Math.max(8, errRadiusX * 0.7);
    ctx.fillStyle = "rgba(34, 197, 94, 0.12)";
    ctx.strokeStyle = "rgba(34, 197, 94, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(toX(spk[0]), toY(spk[1]), errRadiusX, errRadiusY, 0, 0, 2 * Math.PI);
    ctx.fill(); ctx.stroke();

    // Speaker Marker Dot
    ctx.fillStyle = confidenceColor(conf);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(toX(spk[0]), toY(spk[1]), 7, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

    // Coordinate & AoA Tag
    ctx.fillStyle = "#f9fafb";
    ctx.font = "bold 10px Inter, sans-serif";
    const aoaTag = tdoa.angle_of_arrival_deg !== undefined ? ` (AoA: ${tdoa.angle_of_arrival_deg > 0 ? "+" : ""}${tdoa.angle_of_arrival_deg.toFixed(1)}°)` : "";
    ctx.fillText(`🔊 Speaker [${spk[0].toFixed(2)}m, ${spk[1].toFixed(2)}m]${aoaTag}`, toX(spk[0]) + 10, toY(spk[1]) - 4);
  }

  // 8. Physical Equation & Speed of Sound Readout
  ctx.fillStyle = "rgba(11, 17, 32, 0.85)";
  ctx.fillRect(4, 4, 280, 22);
  ctx.strokeStyle = "#374151";
  ctx.lineWidth = 1;
  ctx.strokeRect(4, 4, 280, 22);

  ctx.fillStyle = "#38bdf8";
  ctx.font = "bold 9.5px monospace";
  const cVal = locus.speed_of_sound_mps ? locus.speed_of_sound_mps.toFixed(1) : "344.4";
  ctx.fillText(`||x-m1|| - ||x-m2|| = c(T)·Δt = ${(locus.distance_diff_m * 100).toFixed(1)}cm`, 8, 18);
}

// ─── TDOA Timeline Canvas (Kalman Trajectory + CRLB Uncertainty Band) ─────────
function drawTDOATimeline() {
  const canvas = document.getElementById("hwTimelineCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (tdoaHistory.length < 2) {
    ctx.fillStyle = "#4b5563";
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Waiting for real-time multi-device acoustic packets…", W / 2, H / 2);
    ctx.textAlign = "left";
    return;
  }

  const maxRange = Math.max(4.0, ...tdoaHistory.map(Math.abs), ...rawTdoaHistory.map(Math.abs));
  const toX = i => (i / (MAX_HISTORY - 1)) * W;
  const toY = v => H / 2 - (v / maxRange) * (H * 0.42);

  // Background zones (Laptop vs Phone side)
  ctx.fillStyle = "rgba(56, 189, 248, 0.05)";
  ctx.fillRect(0, 0, W, H / 2); // Upper: Laptop closer (tau > 0)
  ctx.fillStyle = "rgba(34, 197, 94, 0.05)";
  ctx.fillRect(0, H / 2, W, H / 2); // Lower: Phone closer (tau < 0)

  // Zero-delay center axis & gridlines
  [-maxRange, -maxRange / 2, 0, maxRange / 2, maxRange].forEach(v => {
    ctx.strokeStyle = v === 0 ? "#4b5563" : "#1f2937";
    ctx.lineWidth = v === 0 ? 1.5 : 1;
    ctx.setLineDash(v === 0 ? [] : [3, 3]);
    ctx.beginPath(); ctx.moveTo(0, toY(v)); ctx.lineTo(W, toY(v)); ctx.stroke();
    ctx.setLineDash([]);
    if (v !== 0) {
      ctx.fillStyle = "#6b7280";
      ctx.font = "9px monospace";
      ctx.fillText(`${v > 0 ? "+" : ""}${v.toFixed(1)}ms`, 2, toY(v) - 2);
    }
  });

  // Physical Direction Labels
  ctx.fillStyle = "#38bdf8"; ctx.font = "9px Inter, sans-serif"; ctx.fillText("▲ Closer to M1 (Laptop)", 4, 12);
  ctx.fillStyle = "#22c55e"; ctx.fillText("▼ Closer to M2 (Phone)", 4, H - 4);

  // 1. CRLB Uncertainty Ribbon (+/- 1 sigma)
  ctx.fillStyle = "rgba(56, 189, 248, 0.15)";
  ctx.beginPath();
  tdoaHistory.forEach((v, i) => {
    const sig = sigmaHistory[i] || 0.1;
    const yTop = toY(v + sig);
    if (i === 0) ctx.moveTo(toX(i), yTop); else ctx.lineTo(toX(i), yTop);
  });
  for (let i = tdoaHistory.length - 1; i >= 0; i--) {
    const v = tdoaHistory[i];
    const sig = sigmaHistory[i] || 0.1;
    const yBot = toY(v - sig);
    ctx.lineTo(toX(i), yBot);
  }
  ctx.closePath();
  ctx.fill();

  // 2. Raw Sinc-Interpolated Samples (Faint dots)
  rawTdoaHistory.forEach((v, i) => {
    const psr = psrHistory[i] || 1.0;
    if (psr < 1.3) return; // skip noisy invalid frames
    ctx.fillStyle = "rgba(156, 163, 175, 0.4)";
    ctx.beginPath(); ctx.arc(toX(i), toY(v), 2, 0, 2 * Math.PI); ctx.fill();
  });

  // 3. Kalman Filtered Trajectory Curve (Smooth blue line)
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.shadowColor = "#38bdf8";
  ctx.shadowBlur = 4;
  ctx.beginPath();
  tdoaHistory.forEach((v, i) => {
    if (i === 0) ctx.moveTo(toX(i), toY(v)); else ctx.lineTo(toX(i), toY(v));
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // 4. Live Current Value Head
  const lastTdoa = tdoaHistory[tdoaHistory.length - 1];
  const lastPsr = psrHistory[psrHistory.length - 1] || 1.0;
  const lastSigma = sigmaHistory[sigmaHistory.length - 1] || 0.1;

  ctx.fillStyle = qualityColor(lastPsr);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(toX(tdoaHistory.length - 1), toY(lastTdoa), 5, 0, 2 * Math.PI);
  ctx.fill(); ctx.stroke();

  // Live Readout badge in top right
  ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
  ctx.fillRect(W - 130, 4, 126, 18);
  ctx.strokeStyle = "#374151";
  ctx.strokeRect(W - 130, 4, 126, 18);

  ctx.fillStyle = qualityColor(lastPsr);
  ctx.font = "bold 10px monospace";
  ctx.fillText(`Δt = ${lastTdoa > 0 ? "+" : ""}${lastTdoa.toFixed(3)}ms (±${lastSigma.toFixed(2)})`, W - 124, 16);
}

// ─── Gain Calibration Trigger ──────────────────────────────────────────────────
async function triggerGainCalib() {
  try {
    const res = await fetch("/api/hardware/calibrate_gain", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      const gEl = document.getElementById("hw-gain-calib-val");
      if (gEl) gEl.textContent = `${data.mic_gain_offset_db > 0 ? "+" : ""}${data.mic_gain_offset_db.toFixed(1)} dB`;
      alert(`⚖️ Microphone Gain Normalized!\nHardware Sensitivity Offset: ${data.mic_gain_offset_db > 0 ? "+" : ""}${data.mic_gain_offset_db.toFixed(1)} dB\nBoth microphones are now calibrated to 0 dB difference when centered.`);
    } else {
      alert("⚠️ Phone stream not active. Please connect phone first.");
    }
  } catch (err) {
    alert(`Gain calibration error: ${err.message}`);
  }
}

// ─── Acoustic Time Calibration Trigger ────────────────────────────────────────
async function triggerAcousticCalib() {
  const x = parseFloat(document.getElementById("calib-src-x")?.value) || 3.0;
  const y = parseFloat(document.getElementById("calib-src-y")?.value) || 3.0;
  try {
    const res = await fetch("/api/hardware/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_x: x, source_y: y }),
    });
    const data = await res.json();
    if (data.is_valid) {
      alert(`🎯 Acoustic Calibration Successful!\nEstimated Clock Offset: ${data.estimated_hardware_clock_offset_ms.toFixed(2)} ms\nSpeed of Sound: ${data.speed_of_sound_mps} m/s`);
    } else {
      alert(`⚠️ Calibration Inconclusive:\nPeak prominence too low. Ensure sound is playing clearly.`);
    }
  } catch (err) {
    alert(`Calibration request error: ${err.message}`);
  }
}

// ─── Reprocess Trigger ────────────────────────────────────────────────────────
async function triggerReprocess() {
  try {
    const res = await fetch("/api/hardware/reprocess", { method: "POST" });
    const data = await res.json();
    if (data.reprocessed) {
      console.log("Frame reprocessed:", data);
    }
  } catch (err) {
    console.warn("Reprocess error:", err);
  }
}
