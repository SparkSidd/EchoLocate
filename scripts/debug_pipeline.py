"""Debug script to trace the exact pipeline used by run_experiment_01.py."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pyroomacoustics as pra
import yaml

from src.simulation.source_generator import generate_source_signal
from src.simulation.room_simulator import build_mic_array_3d, _add_noise
from src.localization.gcc_phat import gcc_phat, compute_max_tau_from_geometry

with open('configs/experiment_01.yaml') as f:
    cfg = yaml.safe_load(f)

flat_cfg = {'sampling_rate': cfg['room']['sampling_rate'], 'source': cfg['source']}
source_signal, fs = generate_source_signal(flat_cfg)
print(f"Source signal: {len(source_signal)} samples, {len(source_signal)/fs*1000:.1f} ms")
print(f"Source type: {cfg['source']['type']}")

mic_3d, labels = build_mic_array_3d(cfg['microphones']['positions'], cfg['microphones']['height'])
mic_2d = mic_3d[:2, :].T

src_pos = cfg['single_source']['position']
src_3d = np.array([src_pos[0], src_pos[1], cfg['microphones']['height']])
print(f"Source 3D: {src_3d}")

room_dims = cfg['room']['dimensions']
room = pra.ShoeBox(room_dims, fs=fs, materials=pra.Material(1.0),
                   max_order=0, air_absorption=False, ray_tracing=False)
room.add_microphone(mic_3d)
room.add_source(src_3d, signal=source_signal)
room.simulate()
sigs = room.mic_array.signals
print(f"Signal shape: {sigs.shape}")

sigs_noisy = _add_noise(sigs, cfg['stage_a']['noise_level_db'], seed=42)

max_tau = compute_max_tau_from_geometry(mic_2d, cfg['room']['speed_of_sound'], fs, 1.3)
print(f"max_tau = {max_tau}")

r01 = gcc_phat(sigs_noisy[0], sigs_noisy[1], fs=fs, max_tau_samples=max_tau, mic_i=0, mic_j=1)
r02 = gcc_phat(sigs_noisy[0], sigs_noisy[2], fs=fs, max_tau_samples=max_tau, mic_i=0, mic_j=2)
r12 = gcc_phat(sigs_noisy[1], sigs_noisy[2], fs=fs, max_tau_samples=max_tau, mic_i=1, mic_j=2)

print(f"M1-M2: {r01.tau_samples:.2f} samples (theory: 0.00)")
print(f"M1-M3: {r02.tau_samples:.2f} samples (theory: +129.31)")
print(f"M2-M3: {r12.tau_samples:.2f} samples (theory: +129.31)")

# Check theoretical TDOAs
d0 = np.linalg.norm(src_3d[:2] - mic_3d[:2, 0])
d1 = np.linalg.norm(src_3d[:2] - mic_3d[:2, 1])
d2 = np.linalg.norm(src_3d[:2] - mic_3d[:2, 2])
c = cfg['room']['speed_of_sound']
print(f"\nTheory check:")
print(f"  d0={d0:.4f}, d1={d1:.4f}, d2={d2:.4f}")
print(f"  TDOA_01 = {(d0-d1)/c*fs:.2f} samples")
print(f"  TDOA_02 = {(d0-d2)/c*fs:.2f} samples")
print(f"  TDOA_12 = {(d1-d2)/c*fs:.2f} samples")
