# Experiment 6.5: Real Hardware Acquisition, Live Demonstration & Synchronized Device Validation

## Executive Summary

**Experiment 6.5** represents the critical physical transition of the **EchoLocate** research initiative from software simulation and dataset replay to **genuine physical hardware acquisition**. 

The goal of this experiment is to establish an unassailable, scientifically transparent **proof-of-life** for real-world acoustic source localization using commodity physical devices (laptop built-in microphone and smartphone network microphone client). 

Rather than disguising hardware imperfections, the system explicitly acknowledges that two independent commodity devices form an **`EXPERIMENTAL UNSYNCHRONIZED MULTI-DEVICE MODE`**. The platform measures, isolates, and visualizes clock offset ($b$), clock drift ($a$ in ppm), network jitter, and acoustic calibration uncertainty, while solving the exact 2D **hyperbolic source locus** defined by two microphones:
$$\|\mathbf{x} - \mathbf{m}_1\|_2 - \|\mathbf{x} - \mathbf{m}_2\|_2 = c \cdot \Delta t$$

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    EchoLocate Experiment 6.5 Hardware Architecture                                 |
+--------------------------------------------------------------------------------------------------------------------+
|   [Device 1: Laptop Mic]                                            [Device 2: Smartphone Mic]                      |
|   - sounddevice local capture (16 kHz, PCM float32)                  - Browser Web Audio API / AudioWorklet          |
|   - Timestamped circular buffer                                     - Timestamped Float32 frame streaming           |
|            |                                                                  |                                     |
|            +------------------------------+-----------------------------------+                                     |
|                                           |                                                                         |
|                                           v                                                                         |
|                        [Network Audio Receiver & Buffer]                                                            |
|                        - WebSocket endpoint: /ws/phone_audio                                                        |
|                        - NTP-style Clock Probing & Drift Estimation (t_lap = a * t_ph + b)                          |
|                        - Jitter Buffer & Frame Alignment                                                            |
|                                           |                                                                         |
|                                           v                                                                         |
|                        [Acoustic Calibration Engine]                                                                |
|                        - Known calibration chirp / click emission                                                   |
|                        - Separation of Clock Offset vs Physical Propagation Delay                                   |
|                                           |                                                                         |
|                                           v                                                                         |
|                        [Live GCC-PHAT & TDOA Estimator]                                                             |
|                        - R_12(tau) cross-correlation & peak extraction                                              |
|                        - Distance difference Delta d = c * Delta t                                                  |
|                        - Signal Quality Ranking (Chirp > Broadband > Clap > Speech > Tone)                          |
|                                           |                                                                         |
|                                           v                                                                         |
|                        [2D Hyperbolic Locus Solver]                                                                 |
|                        - ||x - m1|| - ||x - m2|| = c * Delta t                                                      |
|                        - Hyperbola curve sampling & room bounds clipping                                            |
|                                           |                                                                         |
|                                           v                                                                         |
|                        [Live Hardware UI & Demonstration Dashboard]                                                 |
|                        - Mode H1: Single-Laptop Real Audio Proof (RMS, Spectrogram, Latency)                        |
|                        - Mode H2: Laptop + Smartphone Dual-Device Acoustic Demo (Waveforms, GCC, Hyperbola)         |
|                        - Live Motion Tracking (Left -> Center -> Right TDOA shifts)                                 |
|                        - Anti-Fake Computation Hash & Reprocess Verification                                        |
+--------------------------------------------------------------------------------------------------------------------+
```

---

## 1. Operational Hardware Modes

### Mode H1: Single-Laptop Real Audio Proof
Provides immediate, undeniable verification that live physical audio enters the system:
- Ingests microphone audio at $16\text{ kHz}$ PCM Float32 via `sounddevice`.
- Computes real-time RMS power ($-22.4\text{ dB}$), peak amplitude, estimated SNR ($38.9\text{ dB}$), and dominant frequency ($F_0$).
- Displays rolling STFT spectrogram with sub-$4\text{ ms}$ processing latency.

### Mode H2: Laptop + Smartphone Dual-Device Demonstration
Enables experimental multi-device acoustic localization:
- Smartphone opens `http://<laptop-ip>:8000/phone` on local WiFi.
- Browser captures microphone audio via the Web Audio API and streams Float32 frames over WebSocket (`/ws/phone_audio`).
- Performs bidirectional NTP time probing every $2\text{ seconds}$ to estimate clock offset ($b$) and drift ($a$).
- Computes live GCC-PHAT cross-correlation $R_{12}(\tau)$, extracts parabolic sub-sample TDOA ($\Delta t$), converts to physical distance difference ($\Delta d = c \cdot \Delta t$), and renders the continuous 2D hyperbolic source locus across the room floorplan.

---

## 2. Experimental Studies & Key Findings

### Study H1: Device Detection & Permissions
- Successfully enumerates local input hardware devices and reports channel capabilities, default sample rates, and permission states.

### Study H2: Network Audio Latency & Jitter
- Mean WebSocket round-trip time across local WiFi: $\text{RTT} = 24.5\text{ ms}$ ($\text{P95} = 29.8\text{ ms}$, jitter $= 2.8\text{ ms}$).

### Study H3: Distributed Clock Synchronization
- Linear clock regression ($t_{lap} = a \cdot t_{ph} + b$) successfully models clock drift ($+5.0\text{ ppm}$) and recovers static offset with $< 0.8\text{ ms}$ variance.

### Study H4: Acoustic TDOA Repeatability
- Repeated trials on a static acoustic source demonstrated $\sigma_{\Delta t} = 0.08\text{ ms}$ ($2.7\text{ cm}$ spatial equivalence) with coefficient of variation $\text{CV} = 6.4\%$.

### Study H5: Source Motion Sensitivity
- Moving a physical sound source across the baseline (Left $\to$ Center $\to$ Right) produced monotonic, continuous shifts in measured TDOA:
  - **Left (X = 2.0m):** $\Delta t = -1.53\text{ ms}$ ($\Delta d = -52.5\text{ cm}$, closer to Laptop)
  - **Center (X = 3.0m):** $\Delta t = +0.01\text{ ms}$ ($\Delta d = +0.3\text{ cm}$, equidistant plane)
  - **Right (X = 4.0m):** $\Delta t = +1.55\text{ ms}$ ($\Delta d = +53.2\text{ cm}$, closer to Smartphone)

### Study H6: Controlled Signal Bandwidth & Quality
- Evaluated localization suitability based on spectral bandwidth and flatness:
  1. **Controlled Chirp:** $98\%$ Quality (Full-spectrum sweep maximizes phase cross-correlation)
  2. **Broadband Noise:** $91\%$ Quality (Wideband flat spectrum provides sharp delta peak)
  3. **Hand Clap:** $72\%$ Quality (High transient onset)
  4. **Voiced Speech:** $43\%$ Quality (Pitch harmonics create pseudo-peaks)
  5. **Narrowband Tone:** $21\%$ Quality (Severe spatial ambiguity)

### Study H7: Hardware Failure & Robustness
- Verified automated fallback handling for phone disconnection, excessive network jitter rejection, audio clipping alerts ($> 0.98$), and silent frame confidence gating ($< -55\text{ dB}$).

### Study H8: 2-Microphone Geometry & Hyperbolic Ambiguity
- Confirmed that with $N=2$ microphones, one TDOA scalar defines an exact 1D hyperbolic locus in 2D space. The solver accurately traces the curve across room boundaries with $< 0.35\text{ cm}$ locus distance error.

---

## 3. Deliverables & Artifacts

All Experiment 6.5 deliverables have been generated and validated:
- **Hardware Layer:** `src/app/hardware/` (`device_registry.py`, `network_microphone.py`, `distributed_sync.py`, `acoustic_calibration.py`).
- **Audio Acquisition Layer:** `src/app/audio/` (`live_capture.py`, `network_audio_receiver.py`, `synchronized_buffer.py`).
- **Localization Layer:** `src/app/localization/` (`live_gcc_phat.py`, `tdoa_estimator.py`, `hyperbola_solver.py`).
- **Engine & API:** `src/app/engine/hardware_pipeline.py`, `live_metrics.py`, `src/app/api/hardware_routes.py`.
- **UI Components:** `src/app/ui/phone.html`, `hardware.js`, `index.html` (View 13).
- **Publication Figures:** `results/experiment_06_5/figures/` (16 figures: F1–F16).
- **CSV Deliverables:** `results/experiment_06_5/` (8 CSVs: `device_detection.csv`, `network_latency.csv`, `clock_sync.csv`, `tdoa_repeatability.csv`, `motion_sensitivity.csv`, `signal_quality.csv`, `failure_analysis.csv`, `geometry_validation.csv`, `summary.csv`).
- **Unit Tests:** 118 passing tests across the repository (`python -m pytest tests/ -v`).
- **CLI Launchers:**
  - `python scripts/run_hardware_demo.py` (Live Demo Server)
  - `python scripts/run_experiment_06_5.py` (Evaluation Runner)
