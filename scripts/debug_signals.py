"""Diagnostic: check how pyroomacoustics returns delayed signals."""
import sys
from pathlib import Path
import numpy as np
import pyroomacoustics as pra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.simulation.source_generator import generate_chirp
from src.simulation.room_simulator import build_mic_array_3d

fs = 16000
c = 343.0

# Room
room_dims = [10.0, 6.0, 3.0]

# Mics
mic_3d, labels = build_mic_array_3d({"M1": [1.0, 1.5], "M2": [9.0, 1.5], "M3": [5.0, 4.5]}, height=1.2)

# Source at centre — should be equidistant from M1 and M2
src = np.array([5.0, 3.0, 1.2])

# Generate chirp
chirp = generate_chirp(fs, duration=0.005, f_low=300, f_high=7000,
                       pad_before=0.01, pad_after=0.15)

print(f"Chirp length: {len(chirp)} samples = {len(chirp)/fs*1000:.1f} ms")
print(f"\nGeometric distances from source ({src[:2]}):")
for label, col in zip(labels, range(3)):
    mic_pos = mic_3d[:2, col]
    d = np.linalg.norm(src[:2] - mic_pos)
    t = d / c
    n_delay = t * fs
    print(f"  {label} at {mic_pos}: d={d:.4f}m, t={t*1000:.4f}ms, n_delay={n_delay:.2f} samples")

# Compute theoretical TDOAs
d0 = np.linalg.norm(src[:2] - mic_3d[:2, 0])
d1 = np.linalg.norm(src[:2] - mic_3d[:2, 1])
d2 = np.linalg.norm(src[:2] - mic_3d[:2, 2])
print(f"\nTheoretical TDOAs:")
print(f"  TDOA_01 = (d0-d1)/c = ({d0:.4f}-{d1:.4f})/343 = {(d0-d1)/c*1000:.4f} ms = {(d0-d1)/c*fs:.2f} samples")
print(f"  TDOA_02 = (d0-d2)/c = {(d0-d2)/c*1000:.4f} ms = {(d0-d2)/c*fs:.2f} samples")
print(f"  TDOA_12 = (d1-d2)/c = {(d1-d2)/c*1000:.4f} ms = {(d1-d2)/c*fs:.2f} samples")

# Simulate
room = pra.ShoeBox(room_dims, fs=fs, materials=pra.Material(1.0), max_order=0,
                   air_absorption=False, ray_tracing=False)
room.add_microphone(mic_3d)
room.add_source(src, signal=chirp)
room.simulate()

sigs = room.mic_array.signals
print(f"\nSimulated signal shapes: {sigs.shape}")
print(f"Signal lengths: M1={len(sigs[0])}, M2={len(sigs[1])}, M3={len(sigs[2])}")

# Find first significant sample in each channel
threshold = 0.001 * np.max(np.abs(sigs))
for i, label in enumerate(labels):
    above = np.where(np.abs(sigs[i]) > threshold)[0]
    if len(above):
        first_sample = above[0]
        first_ms = first_sample / fs * 1000
        print(f"  {label}: first significant sample = {first_sample} ({first_ms:.3f} ms)")
    else:
        print(f"  {label}: no signal detected above threshold")

# Peak amplitudes
for i, label in enumerate(labels):
    peak_idx = np.argmax(np.abs(sigs[i]))
    peak_ms = peak_idx / fs * 1000
    print(f"  {label}: peak at sample {peak_idx} ({peak_ms:.3f} ms), amp={sigs[i, peak_idx]:.6f}")

# Compute actual delays from peaks
peak_0 = np.argmax(np.abs(sigs[0]))
peak_1 = np.argmax(np.abs(sigs[1]))
peak_2 = np.argmax(np.abs(sigs[2]))
print(f"\nPeak-based TDOA estimates:")
print(f"  TDOA_01 from peaks: {(peak_0 - peak_1) / fs * 1000:.4f} ms = {peak_0-peak_1} samples")
print(f"  TDOA_02 from peaks: {(peak_0 - peak_2) / fs * 1000:.4f} ms = {peak_0-peak_2} samples")
print(f"  TDOA_12 from peaks: {(peak_1 - peak_2) / fs * 1000:.4f} ms = {peak_1-peak_2} samples")

print("\n--- Cross-correlation sanity check (direct) ---")
from scipy.signal import correlate
for i, j, label_i, label_j in [(0,1,'M1','M2'), (0,2,'M1','M3'), (1,2,'M2','M3')]:
    corr = correlate(sigs[i], sigs[j], mode='full')
    n = len(sigs[i])
    lags = np.arange(-(n-1), n)
    peak_lag = lags[np.argmax(corr)]
    d_i = np.linalg.norm(src[:2] - mic_3d[:2, i])
    d_j = np.linalg.norm(src[:2] - mic_3d[:2, j])
    theoretical = (d_i - d_j) / c * fs
    print(f"  {label_i}-{label_j}: xcorr peak at lag={peak_lag} samples (theory={theoretical:.2f})")
