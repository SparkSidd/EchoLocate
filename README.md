# EchoLocate

> **Acoustic Event Localization System**  
> A modular, simulation-first framework for estimating where a sound occurred in an indoor environment.

---

## Overview

EchoLocate is a research-grade Python system for **acoustic source localization** in indoor environments. It follows a rigorous science-first philosophy: understand the physics, build a verified simulation, implement a classical baseline, measure it quantitatively — and only then introduce advanced AI/ML methods.

**Current version:** Experiment 2 — Robustness & Microphone Geometry Analysis (completed)

---

## Features

| Component | Description |
|---|---|
| **Room Simulation** | pyroomacoustics ShoeBox model with measured Schroeder $RT_{60}$ distribution |
| **Source Signals** | Linear chirp (300–7000 Hz), click/impulse |
| **TDOA Estimation** | GCC-PHAT with gated frequency whitening |
| **Localization** | Nonlinear Least Squares (NLS) multilateration with random restarts |
| **Robustness Grid (Exp 2A)** | 7 $RT_{60}$ levels $\times$ 6 SNR levels $\times$ 5 trials ($3,150$ evaluations) |
| **Geometry Study (Exp 2B)** | 4 geometries (Baseline, Wide, Compact, Corridor) $\times$ 5 conditions |
| **Visualizations** | 30 publication-quality figures across Exp 1 & Exp 2 |
| **Unit Tests** | 39 pytest unit tests with 100% pass rate |

---

## Experiment 2 Key Results

### Part A: Environmental Robustness (Reverberation Cliff Edge)
- **Anechoic ($RT_{60} = 0.0$ s):** Mean error = **5.28 cm**, Success ($\le 1$ m) = **100%**.
- **Mild Reverb ($RT_{60} = 0.2$ s):** Mean error = **24.76 cm**, Success ($\le 1$ m) = **100%**.
- **Cliff Edge ($RT_{60} = 0.3$ s):** Mean error jumps to **1.47 m**, Success ($\le 1$ m) collapses to **20.0%**.
- **Severe Reverb ($RT_{60} = 0.6$ s):** Mean error = **2.72 m**, Success ($\le 1$ m) = **13.3%**.

### Part B: Microphone Geometry Rankings

| Rank | Geometry | Triangle Area | Overall Mean Error | Success Rate ($\le 1$ m) | Recommendation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Compact Triangle (C)** | **1.0 $\text{m}^2$** | **1.09 m** | **56.0%** | **Recommended for deployment** |
| **2** | Baseline Triangle (A) | 12.0 $\text{m}^2$ | 1.43 m | 49.3% | Optimal only for anechoic rooms |
| **3** | Longitudinal Corridor (D) | 1.05 $\text{m}^2$ | 1.51 m | 29.6% | Requires cross-axis offset |
| **4** | Wide Triangle (B) | 16.0 $\text{m}^2$ | 1.65 m | 38.7% | High reflection vulnerability |

---

## Project Structure

```
EchoLocate/
├── configs/
│   └── experiment_01.yaml      # Room, mic, source, solver settings
├── src/
│   ├── simulation/
│   │   ├── room_simulator.py   # pyroomacoustics wrapper
│   │   └── source_generator.py # Chirp/click generation
│   ├── localization/
│   │   ├── gcc_phat.py         # GCC-PHAT TDOA estimator
│   │   └── multilateration.py  # NLS source localization
│   ├── evaluation/
│   │   └── metrics.py          # TDOA accuracy, localization metrics
│   └── visualization/
│       └── plots.py            # 13 publication-quality figures
├── scripts/
│   └── run_experiment_01.py    # Full experiment orchestrator
├── tests/
│   ├── test_gcc_phat.py        # 11 GCC-PHAT unit tests
│   ├── test_multilateration.py # 11 localization unit tests
│   └── test_simulation.py      # 11 simulation unit tests
├── docs/
│   └── experiment_01_report.md # Full technical report
├── results/
│   └── experiment_01/
│       ├── figures/            # 13 PNG figures
│       └── *.csv               # Numerical results
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+ (tested on 3.13.4)
- Windows, Linux, or macOS

### Installation

```bash
git clone https://github.com/your-username/EchoLocate.git
cd EchoLocate
pip install -r requirements.txt
```

### Run Experiment 1

```bash
# Run all unit tests first
python -m pytest tests/ -v

# Run the full experiment
python scripts/run_experiment_01.py
```

Results are saved to `results/experiment_01/`.

---

## Configuration

All experiment parameters are in [`configs/experiment_01.yaml`](configs/experiment_01.yaml):

```yaml
room:
  dimensions: [10.0, 6.0, 3.0]   # metres
  sampling_rate: 16000            # Hz
  speed_of_sound: 343.0           # m/s

microphones:
  positions:
    M1: [1.0, 1.5]
    M2: [9.0, 1.5]
    M3: [5.0, 4.5]
  height: 1.2

source:
  type: chirp
  f_low: 300
  f_high: 7000
  duration: 0.005   # 5 ms active chirp

stage_a:
  mode: anechoic
  noise_level_db: -60   # 60 dB SNR (very quiet)

stage_b:
  mode: reverberant
  rt60: 0.3             # 300 ms reverberation time
  noise_level_db: -20   # 20 dB SNR
```

---

## Design Philosophy

EchoLocate follows a **simulation-first, physics-first** approach:

1. **Understand the physical signal problem** — propagation delays, reflections, noise
2. **Build a controllable simulation** — before touching real audio
3. **Implement a classical baseline** — GCC-PHAT + multilateration (no ML)
4. **Measure quantitatively** — TDOA accuracy in samples, localization error in metres
5. **Only then consider AI/ML** — as a measured improvement over the verified baseline

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pyroomacoustics` | ≥ 0.8.0 | Room acoustics simulation |
| `numpy` | ≥ 2.0.0 | Signal processing |
| `scipy` | ≥ 1.10.0 | Optimization (NLS solver) |
| `matplotlib` | ≥ 3.7.0 | Visualization |
| `soundfile` | ≥ 0.12.0 | Audio I/O |
| `librosa` | ≥ 0.10.0 | Audio analysis utilities |
| `pyyaml` | ≥ 6.0 | Configuration |
| `pytest` | ≥ 7.4.0 | Unit testing |

---

## Roadmap

- [x] **Experiment 1:** Baseline GCC-PHAT + NLS multilateration (completed)
- [ ] **Experiment 2:** Sub-sample TDOA interpolation + SRP-PHAT
- [ ] **Experiment 3:** Effect of RT60 on localization accuracy (RT60 sweep)
- [ ] **Experiment 4:** Real-world microphone array (hostel corridor prototype)
- [ ] **Experiment 5:** Deep learning TDOA estimation (comparison to baseline)

---

## License

MIT License — see [LICENSE](LICENSE)

---

*EchoLocate — Research prototype. Python 3.13.4. pyroomacoustics 0.10.1.*
