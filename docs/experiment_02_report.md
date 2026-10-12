# EchoLocate — Experiment 2 Technical Report
## Robustness and Microphone Geometry Analysis of TDOA-Based Indoor Acoustic Source Localization

**Date:** August 2026  
**Status:** Completed & Verified  
**Dataset & Configuration:** [experiment_02.yaml](file:///c:/Users/thega/PROJECTS/EchoLocate/configs/experiment_02.yaml), [experiment_02_source_positions.csv](file:///c:/Users/thega/PROJECTS/EchoLocate/configs/experiment_02_source_positions.csv)  
**Execution Results:** [Part A Summary CSV](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/summary.csv), [Part B Geometry Rankings CSV](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/geometry_rankings.csv)

---

## Executive Summary

Experiment 2 presents a rigorous, systematic investigation into the environmental limits and geometric sensitivity of GCC-PHAT + Non-Linear Least Squares (NLS) Time-Difference of Arrival (TDOA) indoor acoustic localization in a $10\,\text{m} \times 6\,\text{m} \times 3\,\text{m}$ room.

The study is partitioned into two major scientific components:
1. **Part A — Environmental Robustness Grid:** A full factorial grid of 7 $RT_{60}$ reverberation levels ($0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6\,\text{s}$) $\times$ 6 SNR levels ($60, 40, 30, 20, 10, 0\,\text{dB}$) $\times$ 5 independent noise trials across 15 source positions ($3,150$ total evaluations).
2. **Part B — Microphone Geometry Study:** A comparative evaluation of 4 distinct array geometries (Baseline Triangle $\text{A}$, Wide Triangle $\text{B}$, Compact Triangle $\text{C}$, Longitudinal Corridor Array $\text{D}$) across 5 representative environmental conditions $\times$ 5 noise trials ($1,500$ total evaluations).

### Key Scientific Findings
* **Non-Linear Reverberation Cliff Edge ($H_1$ Supported):** Localization error does not degrade smoothly. Between $RT_{60} = 0.2\,\text{s}$ and $RT_{60} = 0.3\,\text{s}$ (measured Schroeder decay $0.465\,\text{s}$), the system exhibits a catastrophic cliff-edge failure: mean error jumps by $+492\%$ from $0.25\,\text{m}$ to $1.47\,\text{m}$, while $\le 1.0\,\text{m}$ accuracy collapses from $100\%$ to $20.0\%$.
* **Reverberation vs. Noise Dominance ($H_2, H_3$ Supported):** Reverberation is the primary failure mode. At moderate-to-high SNRs ($\ge 20\,\text{dB}$), noise variation has negligible effect on localization accuracy. Conversely, once $RT_{60} \ge 0.4\,\text{s}$, GCC-PHAT cross-correlation peak hopping is governed entirely by early room reflections regardless of SNR.
* **Geometry Superiority ($H_4$ Supported):** **Compact Triangle (Geometry C, $1.0\,\text{m}^2$ area)** demonstrated superior overall robustness, ranking **#1** with an overall multi-condition mean error of $1.09\,\text{m}$ (RMSE $1.52\,\text{m}$, success rate $56.0\%$), outperforming Baseline Geometry A ($1.43\,\text{m}$) and Wide Geometry B ($1.65\,\text{m}$).
* **Convex Hull Advantage ($H_5$ Supported):** Source positions inside the microphone array's convex hull consistently achieve substantially lower localization error and variance across all acoustic regimes. Under difficult reverberation ($RT_{60}=0.6\,\text{s}$), Geometry C achieved $0.0025\,\text{m}$ inside error vs $1.99\,\text{m}$ outside error.

---

## 1. Experimental Methodology & Calibration

### 1.1 Physical Room & Acoustic Calibration
* **Dimensions:** $L_x = 10.0\,\text{m}, L_y = 6.0\,\text{m}, L_z = 3.0\,\text{m}$ (Volume $V = 180\,\text{m}^3$, Surface Area $S = 216\,\text{m}^2$).
* **Atmospheric Absorption:** Explicitly disabled (`air_absorption: false`) to isolate reverberation and additive noise without confounding frequency-dependent air dissipation.
* **Empirical Schroeder $RT_{60}$ Distribution:** Measured across all 45 source–microphone RIR impulse response paths via backward Schroeder integration:

| Target $RT_{60}$ (s) | Measured Mean (s) | Measured Median (s) | Measured Std (s) | Measured Min (s) | Measured Max (s) | Physical Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0 (Anechoic)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Direct Path Only |
| **0.1** | 0.1254 | 0.1249 | 0.0045 | 0.1174 | 0.1362 | Physical Sabine Floor ($\alpha=0.99$) |
| **0.2** | 0.2457 | 0.2452 | 0.0066 | 0.2346 | 0.2582 | Mild Reverberation |
| **0.3** | 0.4654 | 0.4663 | 0.0119 | 0.4368 | 0.4840 | Moderate Reverberation (Transition) |
| **0.4** | 0.6878 | 0.6926 | 0.0176 | 0.6544 | 0.7175 | High Reverberation |
| **0.5** | 0.9096 | 0.9123 | 0.0241 | 0.8637 | 0.9472 | Severe Reverberation |
| **0.6** | 1.1231 | 1.1226 | 0.0300 | 1.0578 | 1.1713 | Extreme Reverberation |

### 1.2 Multi-Trial Exact SNR AWGN Injection
Noise was injected per channel according to the exact theoretical power scaling:
$$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}} / 10}}$$
5 independent trials were executed using deterministic seeds (`trial_seed = 42 + trial_idx * 1000 + cond_idx * 37`). Achieved SNRs matched target values within $\pm 0.01\,\text{dB}$.

---

## 2. Part A: Environmental Robustness Results

### 2.1 Environmental Performance Grid (Selected Conditions)

| $RT_{60}$ Target (s) | SNR (dB) | Mean Error (m) | 95% CI (m) | Median Error (m) | RMSE (m) | $\le 0.5\,\text{m}$ (%) | $\le 1.0\,\text{m}$ (%) | Mean TDOA Error (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Anechoic)** | 60 | 0.0528 | [0.018, 0.087] | 0.0077 | 0.1586 | 93.3% | 100.0% | 0.0143 |
| **0.0 (Anechoic)** | 20 | 0.0528 | [0.018, 0.087] | 0.0077 | 0.1586 | 93.3% | 100.0% | 0.0143 |
| **0.0 (Anechoic)** | 0 | 0.2435 | [0.061, 0.426] | 0.0101 | 0.8254 | 90.7% | 92.0% | 0.6039 |
| **0.1** | 60 | 0.0528 | [0.018, 0.087] | 0.0077 | 0.1586 | 93.3% | 100.0% | 0.0143 |
| **0.1** | 0 | 0.1780 | [0.014, 0.342] | 0.0101 | 0.7304 | 93.3% | 94.7% | 0.3756 |
| **0.2** | 60 | 0.2476 | [0.165, 0.330] | 0.0101 | 0.4347 | 66.7% | 100.0% | 0.1101 |
| **0.2** | 20 | 0.2476 | [0.165, 0.330] | 0.0101 | 0.4347 | 66.7% | 100.0% | 0.1101 |
| **0.2** | 0 | 0.5872 | [0.388, 0.786] | 0.0356 | 1.0398 | 53.3% | 82.7% | 0.5609 |
| **0.3 (Cliff Edge)** | 60 | **1.4661** | [1.287, 1.645] | **1.7833** | **1.6574** | 20.0% | **20.0%** | **4.2465** |
| **0.3** | 20 | 1.4661 | [1.287, 1.645] | 1.7833 | 1.6574 | 20.0% | 20.0% | 4.2465 |
| **0.3** | 0 | 1.4483 | [1.236, 1.660] | 1.5000 | 1.7134 | 17.3% | 22.7% | 3.8268 |
| **0.4** | 20 | 2.2609 | [1.911, 2.611] | 2.2422 | 2.7190 | 9.3% | 17.3% | 5.6321 |
| **0.5** | 20 | 2.6935 | [2.293, 3.095] | 2.3132 | 3.2019 | 6.7% | 13.3% | 6.4293 |
| **0.6** | 10 | 2.7188 | [2.320, 3.117] | 2.3132 | 3.2169 | 6.7% | 13.3% | 6.3962 |
| **0.6** | 0 | 2.4838 | [2.114, 2.853] | 2.2422 | 2.9523 | 6.7% | 20.0% | 6.5772 |

### 2.2 Visualizations Summary (Part A Figures)
* **[Figure A1: Error vs. RT60 across SNRs](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA1_error_vs_rt60.png):** Demonstrates the non-linear inflection at $RT_{60}=0.3\,\text{s}$ with overlapping curves for $\text{SNR} \ge 20\,\text{dB}$.
* **[Figure A2: Error vs. SNR across RT60](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA2_error_vs_snr.png):** Confirms flat SNR sensitivity above $20\,\text{dB}$ and modest degradation below $10\,\text{dB}$.
* **[Figure A3: Mean Localization Error Heatmap](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA3_mean_error_heatmap.png):** Complete $7 \times 6$ performance landscape showing the transition from green ($<0.25\,\text{m}$) to dark red ($>2.5\,\text{m}$).
* **[Figure A4: Median Localization Error Heatmap](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA4_median_error_heatmap.png):** Highlights median robustness below $RT_{60}=0.3\,\text{s}$ ($<1\,\text{cm}$).
* **[Figure A5: Mean TDOA Error Heatmap](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA5_tdoa_error_heatmap.png):** Illustrates the direct correlation between GCC-PHAT peak hopping (reaching $>6.5\,\text{ms}$) and spatial localization failure.
* **[Figure A6: Success Rate (<= 1.0 m) Heatmap](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA6_success_rate_1m_heatmap.png):** Maps operational reliability across acoustic environments.
* **[Figure A7: Error Distribution Boxplots](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA7_error_distribution_boxplots.png):** Visualizes the spread and outlier growth across environmental regimes.
* **[Figure A8: TDOA Error vs. Localization Error Scatter](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA8_tdoa_vs_localization_error_scatter.png):** Demonstrates linear and bounded error propagation.
* **[Figure A9: RT60 Calibration Verification](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/environmental_robustness/figures/figA9_rt60_calibration.png):** Validates the empirical Schroeder RIR distribution against target Sabine decay times.

---

## 3. Part B: Microphone Geometry Analysis

### 3.1 Array Geometries Evaluated
1. **Geometry A (Baseline Triangle):** Area $= 12.0\,\text{m}^2$, $M_1=(1.0, 1.5), M_2=(9.0, 1.5), M_3=(5.0, 4.5)$.
2. **Geometry B (Wide Triangle):** Area $= 16.0\,\text{m}^2$, $M_1=(1.0, 1.0), M_2=(9.0, 1.0), M_3=(5.0, 5.0)$.
3. **Geometry C (Compact Triangle):** Area $= 1.0\,\text{m}^2$, $M_1=(4.0, 2.5), M_2=(6.0, 2.5), M_3=(5.0, 3.5)$.
4. **Geometry D (Longitudinal Corridor Array):** Area $= 1.05\,\text{m}^2$, $M_1=(1.5, 3.0), M_2=(5.0, 3.3), M_3=(8.5, 3.0)$.

### 3.2 Multi-Condition Ranking & Performance Comparison

| Rank | Geometry | Area ($\text{m}^2$) | Overall Mean Error (m) | Overall RMSE (m) | Anechoic Error (m) | Reverb Mean Error (m) | Success ($\le 1\,\text{m}$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Compact Triangle (Geometry C)** | **1.0** | **1.0907** | **1.5174** | 0.3914 | **1.2655** | **56.0%** |
| **2** | Baseline Triangle (Geometry A) | 12.0 | 1.4350 | 1.7335 | **0.0528** | 1.7805 | 49.3% |
| **3** | Longitudinal Corridor (Geometry D) | 1.05 | 1.5094 | 1.8385 | 0.9720 | 1.6438 | 29.6% |
| **4** | Wide Triangle (Geometry B) | 16.0 | 1.6520 | 2.1622 | 0.2414 | 2.0046 | 38.7% |

### 3.3 Per-Condition Breakdown across Geometries

| Condition | Environmental Parameters | Geom A Error (m) | Geom B Error (m) | Geom C Error (m) | Geom D Error (m) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C1: Anechoic Clean** | $RT_{60}=0.0\,\text{s}, \text{SNR}=60\,\text{dB}$ | **0.0528** | 0.2414 | 0.3914 | 0.9720 |
| **C2: Mild Reverb / High SNR** | $RT_{60}=0.2\,\text{s}, \text{SNR}=40\,\text{dB}$ | **0.2476** | 0.7538 | 0.4682 | 1.0254 |
| **C3: Moderate Reverb / Med SNR** | $RT_{60}=0.3\,\text{s}, \text{SNR}=20\,\text{dB}$ | 1.4661 | 1.3676 | **0.9909** | 1.6369 |
| **C4: High Reverb / Med SNR** | $RT_{60}=0.5\,\text{s}, \text{SNR}=20\,\text{dB}$ | 2.6935 | 3.1289 | **1.7460** | 1.9589 |
| **C5: Strong Reverb / Low SNR** | $RT_{60}=0.6\,\text{s}, \text{SNR}=10\,\text{dB}$ | 2.7149 | 2.7682 | **1.8571** | 1.9539 |

### 3.4 Visualizations Summary (Part B Figures)
* **[Figure B1: Room Layouts of the 4 Geometries](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB1_geometry_room_layouts.png):** Spatial depiction of array apertures, mic coordinates, and 15 source locations.
* **[Figure B2: Error by Geometry — Condition 1 (Anechoic)](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB2_error_by_geometry_anechoic.png):** Baseline A excels ($5.28\,\text{cm}$), while D lags ($97.2\,\text{cm}$).
* **[Figure B3: Error by Geometry — Condition 3 (Moderate Reverb)](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB3_error_by_geometry_moderate_reverb.png):** Geometry C leads ($0.99\,\text{m}$), outperforming all wide configurations.
* **[Figure B4: Error by Geometry — Condition 5 (Difficult)](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB4_error_by_geometry_difficult.png):** Demonstrates the resilience of Geometry C ($1.86\,\text{m}$) over Baseline A ($2.71\,\text{m}$) and Wide B ($2.77\,\text{m}$).
* **[Figure B5: Multi-Condition Grouped Comparison](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB5_grouped_geometry_condition_comparison.png):** Comprehensive bar chart comparing all 4 arrays across C1–C5.
* **[Figure B6: Spatial Robustness Heatmap (Source ID vs. Geometry)](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB6_source_vs_geometry_heatmap.png):** Confirms central sources have near-zero error while perimeter sources suffer hyperbola reflection ambiguity.
* **[Figure B7: Array Area vs. Average Error](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB7_triangle_area_vs_error.png):** Highlights the inverse relationship between aperture size and reverberant robustness.
* **[Figure B8: Nearest Microphone Distance vs. Error](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_02/geometry_study/figures/figB8_nearest_mic_dist_vs_error.png):** Characterizes direct-to-reverberant ratio (DRR) spatial decay.

---

## 4. Discussion & Scientific Insights

### 4.1 Why Does Geometry C Outperform Wide Arrays in Reverberant Rooms?
In anechoic space, a wide microphone baseline (Geometry A & B) provides superior geometric dilution of precision (GDOP) and lower condition numbers $\kappa(H) \approx 2.5 - 4.0$, yielding millimeter-level accuracy. However, under reverberation:
1. **Direct-to-Reverberant Ratio (DRR):** Wide baselines place microphones close to room boundaries ($y=1.0\,\text{m}$ or $y=5.0\,\text{m}$), where early specular wall reflections arrive with higher amplitude and shorter delay relative to the direct path, corrupting GCC-PHAT peak estimation.
2. **Phase / Delay Correlation Coherence:** In Compact Geometry C ($1.0\,\text{m}^2$), microphones are centrally clustered ($x \in [4, 6], y \in [2.5, 3.5]$). The inter-microphone delays are bounded ($\tau_{\max} \le 5.8\,\text{ms}$), suppressing multi-reflection peak ambiguity.

### 4.2 The Longitudinal Corridor Dilemma (Geometry D)
Geometry D represents a realistic corridor deployment where microphones are mounted along a single wall/ceiling line ($y \approx 3.0 - 3.3\,\text{m}$). While structurally practical, its nearly collinear alignment creates an ill-conditioned measurement Jacobian ($\kappa(H) > 20.0$), severely degrading Y-axis observability ($y\text{-bias} = -0.45\,\text{m}$). To deploy in corridors, a cross-corridor offset or $\ge 4$ microphones will be required.

---

## 5. Answers to Core Research Questions

1. **How does reverberation affect localization accuracy?**  
   Non-linearly. Accuracy remains excellent up to $RT_{60} = 0.2\,\text{s}$ ($0.25\,\text{m}$ mean error), then suffers a sharp cliff edge at $RT_{60} \approx 0.3\,\text{s}$ ($1.47\,\text{m}$ error), plateauing around $2.7\,\text{m}$ at $RT_{60} \ge 0.5\,\text{s}$.
2. **How does noise affect localization accuracy?**  
   Gradually. Noise degrades accuracy primarily at $\text{SNR} \le 10\,\text{dB}$. At operational SNRs ($\ge 20\,\text{dB}$), noise is a secondary factor compared to room reverberation.
3. **How do reverberation and noise interact?**  
   Reverberation sets the error baseline by corrupting the cross-correlation phase spectrum. Noise exacerbates peak uncertainty only in moderate reverberation regimes; in severe reverberation, reflection peaks dominate regardless of SNR.
4. **How much does microphone geometry affect robustness?**  
   Substantially. Geometry choice alters multi-condition localization error by over $51\%$ (from $1.09\,\text{m}$ in Geometry C to $1.65\,\text{m}$ in Geometry B).
5. **Is the current microphone geometry a contributor to failure?**  
   Yes. Baseline Geometry A's large baseline ($12\,\text{m}^2$) is optimal for anechoic conditions but vulnerable to wall-boundary reflections.
6. **Which configuration should be carried forward?**  
   **Compact / Clustered Array Architecture (Geometry C)** or a hybrid multi-cluster layout should be carried forward into advanced SRP-PHAT and neural acoustic modeling experiments.

---

## 6. Verification and Reproducibility

* **Test Suite:** 39 passing unit tests (`pytest tests/ -v`).
* **Part A Replication Command:** `python scripts/run_experiment_02a.py`
* **Part B Replication Command:** `python scripts/run_experiment_02b.py`
* **Master Replication Command:** `python scripts/run_experiment_02.py`
