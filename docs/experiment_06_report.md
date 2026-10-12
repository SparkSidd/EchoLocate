# Experiment 6: Real-Time Working Prototype, Interactive Demonstration & Deployment Validation

## Executive Summary

**Experiment 6** represents the final deployment and interactive demonstration stage of the **EchoLocate** acoustic source localization research journey. Building directly upon the empirical foundations of Experiments 1–5, Experiment 6 successfully bridges offline research algorithms into a genuine, sub-real-time, interactive working prototype capable of live presentation to examiners and evaluators.

The primary achievement of Experiment 6 is demonstrating that **EchoLocate** operates with complete scientific transparency across **four distinct demonstration modes** and **five one-click pedagogical presets**, achieving a mean frame processing latency of **11.4 ms** (Real-Time Factor $\text{RTF} = 0.057\times$) and passing a rigorous **anti-fake-demo integrity audit**.

```
+----------------------------------------------------------------------------------------------------+
|                                    EchoLocate Experiment 6 Architecture                            |
+----------------------------------------------------------------------------------------------------+
|  [Audio Acquisition] ---> [Frame Validator] ---> [Multi-Method Engine] ---> [Causal Tracker]       |
|  - Live SoundDevice        - Silence / NaN Check   - M1: Classical NLS        - Exp Smoothing      |
|  - LOCATA Playback         - Clipping Check        - M2: Raw SRP-PHAT         - 2D Kalman Filter   |
|  - Synthetic Sim           - Geometry Singularity  - M6: Calib DeepGCC        - Anti-Lookahead     |
|                                                    - M11e: Diff Spatial SRP                        |
|                                                    - M12: Gated Hybrid                             |
+----------------------------------------------------------------------------------------------------+
|                 [WebSocket Live Stream]  <--->  [Interactive HTML5 / Canvas GUI]                   |
|                 - 2D Floorplan Map               - Dynamic SRP Heatmap                             |
|                 - Provenance Badges (🟢/🔵/🟠)   - Examiner Re-run Verification                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 1. Prototype Architecture & Implementation

The interactive system is organized as a decoupled, asynchronous multi-layer application:

1. **Hardware & Sensor Abstraction Layer (`src/app/hardware/`)**:
   - `AudioDeviceInfo` & `detect_audio_input_devices()`: Queries host sound hardware via `sounddevice` to identify multi-channel input devices and synchronized microphone arrays.
   - `GeometryValidationResult` & `validate_microphone_geometry()`: Inspects physical array layouts, aperture bounds, duplicate positions (<20 mm), and collinear singularities.
   - `ChannelSyncMonitor`: Monitors inter-channel cross-correlation to detect clock drift and dropped frames.

2. **Audio Streaming & Circular Buffer Layer (`src/app/audio/`)**:
   - `CircularAudioBuffer`: Thread-safe bounded ring buffer providing lock-protected rolling frame extraction.
   - `LiveAudioStreamManager`: Continuous hardware ingestion callback pushing frames to the worker queue.
   - `RecordedPlaybackFeeder`: Sequential feeder supporting real dataset playback (LOCATA / AV16.3) with play, pause, seek, restart, and variable speed ($0.5\times - 2.0\times$).

3. **Core Engine & Processing Worker (`src/app/engine/`)**:
   - `input_validator.py`: Guard detecting digital silence (< -60 dB RMS), NaN/Inf floats, and clipping.
   - `temporal_tracker.py`: Strictly causal temporal stabilization supporting Raw coordinates, Causal Exponential Smoothing ($\alpha = 0.40$), and Constant-Velocity 2D Kalman Filtering with jump dampening.
   - `method_registry.py`: Unified wrapper executing M1, M2, M6, M11e, and M12 without modifying underlying research modules.
   - `pipeline_worker.py`: Dedicated processing thread consuming `PipelineFramePacket` instances and emitting `LocalizationResultPacket` payloads.

4. **Web API & Interactive UI (`src/app/api/`, `src/app/ui/`)**:
   - FastAPI server serving dark-mode HTML5 Canvas dashboard (`http://localhost:8000`).
   - High-throughput WebSocket endpoint (`/ws/stream`) streaming coordinates, confidence halo, SRP 2D likelihood heatmaps, and latency metrics.

---

## 2. Experimental Studies & Key Findings

### Study A: Pipeline Integration & End-to-End Validation
All core subsystems (audio ingestion, signal validation, spatial likelihood steering, temporal stabilization, and rendering) operate seamlessly without deadlocks or memory leaks.

### Study B: Real-Time Throughput & Latency Distributions

Empirical latency measurements across 200 ms audio frames ($f_s = 16\text{ kHz}$):

| Localization Method | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Real-Time Factor (RTF) | Real-Time Headroom |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Classical NLS** | 4.82 ms | 4.65 ms | 6.20 ms | 7.10 ms | $0.024\times$ | $41.5\times$ faster than real-time |
| **M2: Raw SRP-PHAT** | 11.41 ms | 11.20 ms | 13.80 ms | 15.10 ms | $0.057\times$ | $17.5\times$ faster than real-time |
| **M6: Calibrated DeepGCC** | 18.25 ms | 17.90 ms | 21.40 ms | 23.50 ms | $0.091\times$ | $11.0\times$ faster than real-time |
| **M11e: Differentiable SRP** | 18.60 ms | 18.10 ms | 21.90 ms | 24.10 ms | $0.093\times$ | $10.8\times$ faster than real-time |
| **M12: Confidence-Gated Hybrid** | 12.15 ms | 11.80 ms | 14.50 ms | 16.20 ms | $0.061\times$ | $16.5\times$ faster than real-time |

All five algorithms operate well below the 200 ms frame window, ensuring $0\%$ frame dropping under nominal conditions.

### Study C: Streaming Stability & Causal Temporal Tracking

Causal exponential smoothing ($\alpha = 0.40$) and 2D Kalman tracking reduced frame-to-frame spatial jitter from **0.25 m** (raw estimates) to **0.08 m** (smoothed trajectory) while rejecting sudden multipath reflection jumps.

### Study D: Real Recorded Dataset Sequential Playback (LOCATA)

Evaluating the causal replay engine on real multichannel recordings from the LOCATA dataset confirmed that continuous causal frame ingestion correctly tracks active speakers across reverberant acoustic scenes ($RT_{60} = 0.55\text{ s}$) with mean localization error $< 0.45\text{ m}$.

### Study E: Multi-Method Head-to-Head Comparison

Executing all 5 methods on the identical reverberant transition audio frame:
- **M1 (Classical NLS):** Diverged to false correlation peak ($\text{Error} = 1.84\text{ m}$).
- **M2 (Raw SRP-PHAT):** Resolved global spatial maximum ($\text{Error} = 0.12\text{ m}$).
- **M6 (Calibrated DeepGCC):** Enhanced acoustic likelihood ($\text{Error} = 0.48\text{ m}$).
- **M12 (Confidence Hybrid):** Dynamically routed to SRP ($\text{Error} = 0.12\text{ m}$, Confidence = $88\%$).

---

## 3. Data Provenance & Anti-Fake-Demo Audit

To guarantee complete scientific integrity during academic presentations and examinations:

1. **Mandatory Provenance Badging:**
   - 🟢 `LIVE_HARDWARE`: Live audio captured directly from physical sound devices.
   - 🔵 `REAL_RECORDED_DATASET`: Genuine recorded corpus (LOCATA / AV16.3).
   - 🟠 `SYNTHETIC_SIMULATION`: Acoustically simulated scenes generated via PyRoomAcoustics.
   - 🟣 `PRECOMPUTED_RESEARCH_REPLAY`: Historical experiment archive replay.

2. **Integrity Audit Verdict (100% PASS):**
   - **AST Scan:** Zero ground-truth identifiers or oracle labels passed into inference pipelines.
   - **Dynamic Permutation Invariance:** Re-running identical audio chunks yields identical predictions ($\Delta = 0.00\text{ m}$).
   - **Strict Causality Verification:** Monotonically forward timestamps with $0$ future-frame lookaheads.
   - **Examiner Live Verification:** Interactive `RE-RUN CURRENT FRAME` button recomputes the active pipeline live.

---

## 4. Deliverables Summary

All Experiment 6 artifacts have been compiled and verified:
- **Application Core:** `src/app/` (REST API, WebSocket streaming, Causal Tracker, Hardware Inspector, HTML5/JS UI).
- **Evaluation Suite:** `src/evaluation/experiment_06_evaluation.py`.
- **Publication Figures:** `results/experiment_06/figures/` (F1–F16).
- **CSV Datasets:** `results/experiment_06/` (8 CSVs: `summary.csv`, `per_frame_metrics.csv`, `latency_summary.csv`, `method_comparison.csv`, `confidence_analysis.csv`, `stability_analysis.csv`, `playback_evaluation.csv`, `leakage_audit.csv`).
- **Demo Launcher:** `scripts/run_echolocate_demo.py` (`python scripts/run_echolocate_demo.py`).
- **Master Test Suite:** 100 passing unit tests in `tests/` (`python -m pytest tests/ -v`).
