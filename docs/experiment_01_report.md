# EchoLocate — Experiment 01 Report

## Acoustic Event Localization: Baseline TDOA Pipeline

**Date:** 2026-08-26  
**Python Version:** 3.13.4  
**pyroomacoustics version:** 0.10.1

---

## 1. Objective

Establish a clean, reproducible software baseline for acoustic source localization using:
- One indoor shoebox room (10 × 6 × 3 m)
- One sound source (linear chirp, 300–7000 Hz, 5 ms)
- Three microphones in a triangular array
- GCC-PHAT TDOA estimation
- Nonlinear Least Squares (NLS) multilateration
- Two stages: **Stage A** (anechoic, direct path only) and **Stage B** (reverberant, RT60 ≈ 300 ms)

---

## 2. Experimental Setup

### Room Geometry

| Parameter | Value |
|-----------|-------|
| Dimensions | 10.0 × 6.0 × 3.0 m |
| Speed of sound | 343.0 m/s |
| Sampling rate | 16,000 Hz |
| Simulation | pyroomacoustics ShoeBox model |

### Microphone Array (Triangular)

| Mic | Position (x, y) | Height |
|-----|-----------------|--------|
| M1  | (1.00, 1.50) m  | 1.2 m  |
| M2  | (9.00, 1.50) m  | 1.2 m  |
| M3  | (5.00, 4.50) m  | 1.2 m  |

Max baseline (M1–M2): 8.00 m → τ_max ≈ ±486 samples

### Source Signal

| Parameter | Value |
|-----------|-------|
| Type | Linear frequency-swept chirp |
| Duration | 5 ms (active), 165 ms total (with padding) |
| Frequency range | 300–7000 Hz |
| Pre-pad | 10 ms (silence before chirp) |
| Post-pad | 150 ms (silence after chirp) |

### Stage A: Anechoic

| Parameter | Value |
|-----------|-------|
| Mode | `max_order=0` (image source method, direct path only) |
| Noise | −60 dB relative to signal power (SNR = +60 dB) |
| Air absorption | Off |

### Stage B: Reverberant

| Parameter | Value |
|-----------|-------|
| Target RT60 | 300 ms |
| Noise | −20 dB relative to signal power (SNR = +20 dB) |
| Air absorption | On |

### Evaluation Grid

15 source positions evaluated in each stage:
```
(2.0, 1.0), (8.0, 1.0), (5.0, 1.5),
(2.0, 3.0), (5.0, 3.0), (8.0, 3.0),
(2.0, 5.0), (5.0, 5.0), (8.0, 5.0),
(1.5, 1.5), (8.5, 1.5), (1.5, 4.5),
(8.5, 4.5), (1.5, 4.5), (8.5, 4.5)
```

---

## 3. Algorithm: GCC-PHAT + NLS Multilateration

### 3.1 GCC-PHAT (Generalised Cross-Correlation with Phase Transform)

For microphone pair (i, j), the TDOA is estimated as:

```
G_ij(f) = X_i(f) · conj(X_j(f))
Ψ_ij(f) = G_ij(f) / |G_ij(f)|   [PHAT weighting]
τ̂_ij = argmax_τ { IFFT[Ψ_ij(f)] }
```

**Implementation details:**
- FFT size: next power-of-2 of 2×N (avoids circular aliasing)
- Gated PHAT: only whitens frequency bins where geometric mean signal power ≥ 1% of peak (prevents noise from dominating when signals are nearly identical)
- Search window: restricted to ±τ_max = max_baseline / c × 1.3 safety factor
- fftshift used for correct signed-lag extraction

**Sign convention:** τ̂_ij > 0 means mic i is farther from source than mic j (consistent with τ = (d_i − d_j)/c)

### 3.2 Nonlinear Least Squares (NLS) Multilateration

Given TDOAs τ̂_ij for all pairs, solve for source position **p** = (x, y):

```
minimize Σ_(i,j) [ (||p - m_i|| - ||p - m_j||)/c - τ̂_ij ]²
subject to: 0 ≤ x ≤ room_width, 0 ≤ y ≤ room_depth
```

**Implementation:** `scipy.optimize.least_squares` with TRF (Trust Region Reflective) method, 9 random restarts + grid initialization, room boundary constraints.

---

## 4. Results

### 4.1 Stage A: Anechoic

**Single source at (5.0, 3.0):**

| Pair | Theory (s) | Estimated (s) | |Error| (m) |
|------|-----------|---------------|------------|
| M1–M2 | +0.000000 | +0.000000 | 0.0000 |
| M1–M3 | +0.008082 | +0.008063 | 0.0065 |
| M2–M3 | +0.008082 | +0.008063 | 0.0065 |

**Localization:** True=(5.000, 3.000), Predicted≈(5.000, 3.000), Error: **< 1 cm** ✅

**Multi-source performance (15 positions):**

| Metric | Value |
|--------|-------|
| Mean Error | **0.0528 m** (5.3 cm) |
| Median Error | **0.0077 m** (7.7 mm) |
| RMSE | **0.1586 m** (15.9 cm) |
| Within 0.5 m | **93.3%** |
| Within 1.0 m | **100.0%** |
| Within 2.0 m | **100.0%** |

### 4.2 Stage B: Reverberant (RT60 ≈ 300 ms)

**Multi-source performance (15 positions):**

| Metric | Value |
|--------|-------|
| Mean Error | **2.07 m** |
| Median Error | **2.23 m** |
| RMSE | **2.54 m** |
| Within 0.5 m | **6.7%** |
| Within 1.0 m | **20.0%** |
| Within 2.0 m | **46.7%** |
| Min Error | **0.0049 m** |
| Max Error | **5.35 m** |

### 4.3 Performance Comparison

| Stage | Mean Error | Median Error | RMSE | Within 0.5m | Within 1.0m |
|-------|-----------|--------------|------|-------------|-------------|
| A: Anechoic | 0.053 m | 0.008 m | 0.159 m | 93.3% | **100.0%** |
| B: Reverberant | 2.070 m | 2.234 m | 2.544 m | 6.7% | 20.0% |
| **Degradation** | **39×** | **290×** | **16×** | — | — |

---

## 5. Analysis

### 5.1 Anechoic Stage — Excellent Performance

The anechoic GCC-PHAT achieves near-perfect TDOA estimation. Median error is only 7.7 mm — well below one sample (1/16000 Hz × 343 m/s = 21.4 mm/sample). The sub-sample accuracy is possible because the chirp's spectral richness creates a sharp GCC-PHAT correlation peak.

The few outlier cases (RMSE > median) occur when the source lies close to the symmetry axes of the mic array (ambiguous hyperbolas), causing the NLS solver to converge to a geometrically equivalent but incorrect solution.

### 5.2 Reverberant Stage — Expected Degradation

Reverberation (RT60 ≈ 300 ms) severely degrades TDOA estimation. Key phenomena:

1. **Reflection interference:** Early reflections (< 20 ms) arrive near the direct path and create competing correlation peaks in GCC-PHAT
2. **Biased TDOA estimates:** For several sources, GCC-PHAT identifies reflection peaks as the dominant peak, causing errors of 3–10 m in TDOA space
3. **Some sources localize well:** Min error = 4.9 mm, proving that even in reverberant conditions, certain source positions (particularly near mic array center) can be localized accurately
4. **Tau=0 bias:** Several reverberant TDOAs collapse to 0 (suggesting equal-strength reflections cancelling the direct path phase)

### 5.3 Bugs Found and Fixed

During execution, three significant bugs were identified and corrected:

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| **ShoeBox API error** | `pyroomacoustics` 0.10.1 no longer accepts `c=` parameter | Removed; default 20°C gives c=343.0 m/s |
| **Noise sign convention** | `noise_level_db=-60` was computing noise 60dB *above* signal (formula inverted) | Fixed: `noise_power = signal_power × 10^(snr/10)` |
| **GCC-PHAT peak extraction** | Old IRFFT + roll approach produced wrong signed lag | Replaced with `fftshift` + proper `tau_samples_axis` array |

### 5.4 Limitations of This Baseline

1. **No sub-sample interpolation:** TDOA resolution is limited to 1 sample = 21.4 mm at 16kHz. Parabolic interpolation could improve this.
2. **No PHAT weighting pre-filter:** A Wiener pre-filter or bandpass gate would improve reverberant performance
3. **Static mic array:** The triangular array has poor angular resolution along the M1–M2 baseline axis
4. **Single source:** No multi-source separation or interference rejection

---

## 6. Post-Experiment Analysis & Validation

Following completion of Experiment 1, a detailed post-experiment validation was conducted across all 15 source positions and 45 acoustic propagation paths.

### 6.1 RT60 Empirical Verification (Schroeder Method)

The room simulation used pyroomacoustics' `inverse_sabine(rt60, room_dims)` to set wall absorption coefficients for a target $RT_{60} = 0.400$ s. To verify the actual physical decay in the synthesized Room Impulse Responses (RIRs), the empirical reverberation time was measured across all 45 source-microphone RIRs using Schroeder backward integration (`pra.experimental.measure_rt60`):

| Metric | Target Value | Empirical Measured Value | Difference |
|---|---|---|---|
| **Mean $RT_{60}$** | 0.4000 s | **0.6710 s** | +0.2710 s (+67.8%) |
| **Median $RT_{60}$** | 0.4000 s | **0.6741 s** | +0.2741 s (+68.5%) |
| **Min $RT_{60}$** | 0.4000 s | **0.6033 s** | +0.2033 s |
| **Max $RT_{60}$** | 0.4000 s | **0.7202 s** | +0.3202 s |
| **Standard Deviation** | — | **0.0201 s** | — |

> [!NOTE]
> **Why Measured $RT_{60}$ Exceeds Target:** Sabine's classical formula ($RT_{60} \approx 0.161 V / A$) assumes a perfectly diffuse, isotropic acoustic field. In an image-source shoebox simulation with discrete wall reflections, air absorption, and non-uniform acoustic energy decay across axial/tangential modes, Sabine's equation systematically underestimates the actual reverberation time (or under-prescribes absorption). Consequently, the actual simulated acoustic environment in Stage B is moderately more reverberant (~0.67s) than the nominal 0.40s target.

Full path-by-path measurements are archived in [`results/experiment_01/reverberant_rt60_measurements.csv`](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_01/reverberant_rt60_measurements.csv).

---

### 6.2 Anechoic Error Distribution & The S09 Singularity

In Stage A (Anechoic), the overall mean error is 5.3 cm, but the median error is **7.7 mm** and the 75th percentile is **1.01 cm**. The disparity between the median (7.7 mm) and RMSE (15.9 cm) is explained by a single structural outlier:

- **Best Localized Source:** S12 at (3.5, 4.0) — Error: **2.1 mm**
- **Typical Sources (14 / 15):** Error $\le$ **3.56 cm**
- **Sole Outlier (1 / 15):** S09 at (5.0, 5.0) — Error: **61.08 cm**

#### Mathematical Cause of the S09 Singularity

Source S09 is located at $(5.0, 5.0)$ m, placing it on the vertical symmetry axis of the room ($x = 5.0$) directly behind microphone M3 at $(5.0, 4.5)$.
1. Because $M_1 = (1.0, 1.5)$ and $M_2 = (9.0, 1.5)$, any point on $x = 5.0$ yields $TDOA(M_1, M_2) = 0.0$ s.
2. Along the line $x = 5.0$, the remaining two TDOA constraints $TDOA(M_1, M_3)$ and $TDOA(M_2, M_3)$ become identical:
   $$f(y) = \frac{1}{c}\left(\sqrt{16 + (y - 1.5)^2} - |y - 4.5|\right) = 0.014038\text{ s}$$
3. The function $f(y)$ has a non-differentiable cusp at $y = 4.5$ (location of M3) and is non-monotonic:
   - At $y = 5.0000$ (true location): $f(5.0) = \frac{1}{c}(5.3151 - 0.5000) = 0.014038$ s
   - At $y = 4.3839$ (folded mirror location): $f(4.3839) = \frac{1}{c}(4.9312 - 0.1161) = 0.014038$ s
4. Both $y = 5.0000$ and $y = 4.3839$ produce **mathematically identical TDOAs** across all 3 microphones. 

When NLS multilateration was run using **perfect, zero-error theoretical TDOAs**, it converged to $(5.000, 4.384)$ with a residual cost of $0.0$, producing an error of **61.61 cm**. This proves that the S09 error is **100% geometric fold ambiguity (hyperbolic singularity)** rather than TDOA estimation error.

---

### 6.3 Separation of TDOA Error vs. Geometry / Solver Error

To separate signal-processing errors from geometric / solver limitations, multilateration was evaluated across all 15 sources under two conditions:
1. **Case 1 (Theoretical TDOA):** Perfect, zero-error analytical TDOAs input to NLS solver.
2. **Case 2 (GCC-PHAT TDOA):** Estimated GCC-PHAT TDOAs input to NLS solver.

| Source ID | True (X, Y) | Error with Perfect TDOA | Error with GCC-PHAT TDOA | Difference ($\Delta$) | Mean TDOA Error |
|---|---|---|---|---|---|
| **S01** | (2.0, 1.0) | < 0.0001 mm | 1.01 cm | 1.01 cm | 0.014 ms (4.7 mm) |
| **S02** | (8.0, 1.0) | < 0.0001 mm | 1.01 cm | 1.01 cm | 0.014 ms (4.7 mm) |
| **S03** | (5.0, 1.0) | < 0.0001 mm | 0.55 cm | 0.55 cm | 0.009 ms (3.2 mm) |
| **S04** | (2.0, 3.0) | < 0.0001 mm | 0.61 cm | 0.61 cm | 0.021 ms (7.1 mm) |
| **S05** | (8.0, 3.0) | < 0.0001 mm | 0.61 cm | 0.61 cm | 0.021 ms (7.1 mm) |
| **S06** | (5.0, 3.0) | < 0.0001 mm | 0.49 cm | 0.49 cm | 0.013 ms (4.4 mm) |
| **S07** | (2.0, 5.0) | 0.07 mm | 2.38 cm | 2.37 cm | 0.011 ms (3.9 mm) |
| **S08** | (8.0, 5.0) | 0.52 mm | 2.41 cm | 2.36 cm | 0.011 ms (3.9 mm) |
| **S09** | (5.0, 5.0) | **61.61 cm** | **61.08 cm** | **-0.53 cm** | 0.016 ms (5.6 mm) |
| **S10** | (3.5, 2.0) | < 0.0001 mm | 0.77 cm | 0.77 cm | 0.016 ms (5.4 mm) |
| **S11** | (6.5, 2.0) | < 0.0001 mm | 0.77 cm | 0.77 cm | 0.016 ms (5.4 mm) |
| **S12** | (3.5, 4.0) | 0.002 mm | 0.21 cm | 0.21 cm | 0.007 ms (2.4 mm) |
| **S13** | (6.5, 4.0) | 0.008 mm | 0.21 cm | 0.21 cm | 0.007 ms (2.4 mm) |
| **S14** | (1.5, 4.5) | 0.009 mm | 3.56 cm | 3.56 cm | 0.019 ms (6.6 mm) |
| **S15** | (8.5, 4.5) | 0.023 mm | 3.56 cm | 3.56 cm | 0.019 ms (6.6 mm) |

**Conclusion:** For 14 out of 15 positions, NLS multilateration with perfect TDOA achieves sub-millimeter precision (< 0.5 mm). The entire baseline error in typical anechoic positions is driven by the sample quantization limit of GCC-PHAT at 16 kHz ($\pm 0.23$ samples $\approx 4.9$ mm). For S09, the error is 100% geometric ambiguity.

Full data is archived in [`results/experiment_01/tdoa_vs_solver_error_separation.csv`](file:///c:/Users/thega/PROJECTS/EchoLocate/results/experiment_01/tdoa_vs_solver_error_separation.csv).

---

### 6.4 Reverberant Failure Mechanisms (Signal Processing Evidence)

Inspection of the GCC-PHAT correlation functions in Stage B reveals three distinct failure modes:

1. **Reflection Peak Dominance (False Direct Path):** In corners and near-wall positions (e.g. S14, S15), early wall reflections constructively interfere, producing secondary peaks higher than the direct-path arrival. GCC-PHAT's `argmax` selects a reflection peak, introducing discrete delay errors of 4–13 ms (1.5–4.5 m).
2. **Correlation Peak Broadening & Phase Smearing:** Multiple overlapping room modes and late reverberation whiten out the sharp delta peak of GCC-PHAT, shifting the maximum by several samples even when the direct path is nominally dominant.
3. **Symmetry Collapse ($\tau \approx 0$ Bias):** For sources equidistant from reflections, early echo symmetry cancels phase differences, biasing the estimated TDOA toward zero.

---

### 6.5 Systematic Bias Analysis

| Stage | Mean X Bias | Std X | Mean Y Bias | Std Y |
|---|---|---|---|---|
| **Stage A (Anechoic)** | -0.0000 m | 0.0090 m | -0.0405 m | 0.1585 m |
| **Stage B (Reverberant)** | -0.0001 m | 1.6362 m | **-1.0875 m** | 1.7287 m |

- **X-Axis:** Due to array symmetry ($M_1$ at $x=1$, $M_2$ at $x=9$, $M_3$ at $x=5$), the system has zero systematic X bias in both stages.
- **Y-Axis:** In reverberation, the system exhibits a strong **downward systematic bias** ($\bar{\mu}_Y = -1.09$ m). When reverberation collapses TDOA estimates toward 0, hyperbolic multilateration pulls predicted coordinates inward toward the baseline connecting $M_1$ and $M_2$ ($y = 1.5$ m).

---

## 7. Generated Figures

### Experiment 1 Baseline Figures
| Figure | Description |
|---|---|
| `fig1_room_layout_anechoic.png` | Room geometry + mic array + source position (anechoic) |
| `fig1_room_layout_reverberant.png` | Room geometry + mic array + source position (reverberant) |
| `fig2_waveforms_anechoic.png` | 3-channel waveforms showing time-of-arrival differences |
| `fig2_waveforms_reverberant.png` | 3-channel waveforms with reverberant tails |
| `fig3_waveforms_zoom_anechoic.png` | Zoomed view of direct-path signal arrivals |
| `fig3_waveforms_zoom_reverberant.png` | Zoomed reverberant arrivals |
| `fig4_gcc_phat_anechoic.png` | GCC-PHAT correlation functions for all 3 pairs |
| `fig4_gcc_phat_reverberant.png` | GCC-PHAT correlation functions under reverberation |
| `fig5_localization_single_anechoic.png` | Single-source localization result |
| `fig5_localization_single_reverberant.png` | Single-source reverberant result |
| `fig6_localization_multi_anechoic.png` | Multi-source grid: true vs predicted positions |
| `fig6_localization_multi_reverberant.png` | Multi-source reverberant results |
| `fig7_error_comparison.png` | Anechoic vs reverberant error comparison |

### Post-Analysis Diagnostic Figures (`results/experiment_01/figures/post_analysis/`)
| Figure | Description |
|---|---|
| `figA_error_histogram_anechoic.png` | Histogram of anechoic localization errors |
| `figB_error_cdf_anechoic.png` | Cumulative Distribution Function (CDF) of anechoic errors |
| `figC_room_error_heatmap_anechoic.png` | Room spatial map with sources colored by error |
| `figD_true_vs_pred_vectors_anechoic.png` | True vs predicted source positions with error vectors |
| `figE_error_vs_nearest_mic_dist.png` | Localization error vs distance to nearest microphone |
| `figF_error_vs_room_center_dist.png` | Localization error vs distance from room center |
| `figG_error_vs_geometry_gdop.png` | Localization error vs GDOP & inside/outside array triangle |
| `figH_perfect_vs_estimated_tdoa_localization.png` | Localization error: Perfect TDOA vs GCC-PHAT TDOA |
| `figI_gcc_phat_reverberant_failure_S06.png` | GCC-PHAT response: Anechoic vs Reverberant (Good case: Centre) |
| `figI_gcc_phat_reverberant_failure_S07.png` | GCC-PHAT response: Anechoic vs Reverberant (Median case: Corner) |
| `figI_gcc_phat_reverberant_failure_S15.png` | GCC-PHAT response: Anechoic vs Reverberant (Bad case: Near-wall) |
| `figJ_systematic_bias_scatter.png` | Systematic bias scatter: True vs Predicted X and Y coordinates |

---

## 8. Updated Scientific Conclusions

1. **Direct-Path Localization is Validated:** The GCC-PHAT + NLS multilateration pipeline is mathematically and numerically sound. In anechoic conditions, median localization accuracy is **7.7 mm** (sub-sample precision), and 100% of positions are localized within 1.0 m.
2. **Anechoic Error Decomposition:** Typical anechoic error (mean $\sim 1.3$ cm across 14/15 positions) is governed purely by sample discretization (21.4 mm/sample at 16 kHz). The sole outlier (S09 at 61 cm) is caused by a 3-microphone geometric singularity on the symmetry axis behind M3, not algorithm error.
3. **Reverberation is the Dominant Failure Driver:** Room reverberation ($RT_{60} \approx 0.67$ s) increases mean TDOA error from **0.014 ms to 4.79 ms** (a $340\times$ increase). This causes mean localization error to degrade from **5.3 cm to 2.07 m**.
4. **Primary Failure Mode:** GCC-PHAT's peak-picking mechanism is corrupted by early wall reflections that create secondary correlation peaks exceeding the direct path arrival.
5. **Systematic Inward Bias:** Reverberant TDOA errors cause predictions to collapse toward the microphone baseline ($y \approx 1.5$ m), creating a systematic Y-bias of $-1.09$ m.

---

## 9. Next Steps for Experimentation

Based on the empirical evidence gathered in this validation:
- **Do not jump directly to Deep Learning:** The core physical bottleneck has been isolated to **peak corruption in the cross-correlation domain from early reflections**.
- **Next Logical Step (Experiment 2):** Implement and evaluate **Steered Response Power with Phase Transform (SRP-PHAT)** and **Sub-sample parabolic TDOA interpolation**. SRP-PHAT searches over the entire spatial grid directly, accumulating spatial likelihoods across all pairs rather than committing to discrete pairwise TDOA peak decisions.

---

*Report updated: 2026-08-27 | EchoLocate v0.1.0 | Python 3.13.4*
