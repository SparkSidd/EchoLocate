"""
EchoLocate — Preset Demonstration Scenario Manager
==================================================
Manages 5 one-click demonstration presets with genuine pipeline audio synthesis
and execution for examiner / evaluator presentations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.app.audit.data_provenance import DataProvenance, ProvenanceMetadata
from src.datasets.real_dataset_adapter import LOCATADatasetAdapter
from src.evaluation.cross_domain_analysis import generate_adversarial_speech_utterance
from src.evaluation.neural_robustness_audit import simulate_room_acoustics
from src.experiments.environment_robustness import generate_exact_snr_noise

logger = logging.getLogger(__name__)


@dataclass
class DemoScenarioConfig:
    """Preset scenario metadata and configuration."""
    scenario_id: str
    title: str
    description: str
    expected_outcome: str
    provenance: DataProvenance
    recommended_methods: list[str]
    room_dimensions_m: list[float]
    mic_positions_2d: list[list[float]]
    true_source_xy: tuple[float, float]
    rt60_target_s: float
    snr_db: float
    signal_type: str


class PresetScenarioManager:
    """Provides genuine audio payloads for the 5 preset demonstration scenarios."""

    SCENARIOS: dict[str, DemoScenarioConfig] = {
        "demo_1_stable": DemoScenarioConfig(
            scenario_id="demo_1_stable",
            title="Demo 1: Stable Acoustic Localization",
            description="Low reverberation (RT60 = 0.20s), high SNR (25 dB), broadband chirp.",
            expected_outcome="All methods succeed with high confidence (<0.05m error).",
            provenance=DataProvenance.SYNTHETIC_SIMULATION,
            recommended_methods=["M1_Classical_NLS", "M2_Raw_SRP", "M12_Confidence_Hybrid"],
            room_dimensions_m=[6.0, 4.5, 2.8],
            mic_positions_2d=[[1.5, 1.0], [4.5, 1.0], [3.0, 3.5], [3.0, 1.0]],
            true_source_xy=(3.2, 2.2),
            rt60_target_s=0.20,
            snr_db=25.0,
            signal_type="linear_chirp",
        ),
        "demo_2_classical_failure_srp_recovery": DemoScenarioConfig(
            scenario_id="demo_2_classical_failure_srp_recovery",
            title="Demo 2: Classical Failure & SRP Recovery",
            description="Reverberation transition zone (RT60 = 0.38s), competing reflections.",
            expected_outcome="Classical NLS hops to false correlation peak (>1.5m error); SRP-PHAT recovers true location (<0.20m).",
            provenance=DataProvenance.SYNTHETIC_SIMULATION,
            recommended_methods=["M1_Classical_NLS", "M2_Raw_SRP"],
            room_dimensions_m=[10.0, 6.0, 3.0],
            mic_positions_2d=[[2.0, 1.5], [8.0, 1.5], [5.0, 4.5], [5.0, 1.5]],
            true_source_xy=(4.5, 3.8),
            rt60_target_s=0.38,
            snr_db=18.0,
            signal_type="male_speech",
        ),
        "demo_3_speech_harmonic_ambiguity": DemoScenarioConfig(
            scenario_id="demo_3_speech_harmonic_ambiguity",
            title="Demo 3: Voiced Speech Harmonic Ambiguity",
            description="Strong pitch-periodic harmonic structure (F0 = 140 Hz).",
            expected_outcome="Raw GCC generates false pitch sidelobes; Calibrated DeepGCC suppresses harmonic pseudo-peaks.",
            provenance=DataProvenance.SYNTHETIC_SIMULATION,
            recommended_methods=["M2_Raw_SRP", "M6_Calibrated_DeepGCC", "M12_Confidence_Hybrid"],
            room_dimensions_m=[10.0, 6.0, 3.0],
            mic_positions_2d=[[2.0, 1.5], [8.0, 1.5], [5.0, 4.5], [5.0, 1.5]],
            true_source_xy=(6.2, 2.5),
            rt60_target_s=0.35,
            snr_db=20.0,
            signal_type="voiced_speech",
        ),
        "demo_4_physical_mismatch": DemoScenarioConfig(
            scenario_id="demo_4_physical_mismatch",
            title="Demo 4: Physical Array Mismatch & Uncertainty",
            description="Uncalibrated 10 mm physical microphone misplacement.",
            expected_outcome="Localization degrades (+0.8m error); system drops confidence and triggers ambiguity warning.",
            provenance=DataProvenance.SYNTHETIC_SIMULATION,
            recommended_methods=["M2_Raw_SRP", "M12_Confidence_Hybrid"],
            room_dimensions_m=[10.0, 6.0, 3.0],
            mic_positions_2d=[[2.01, 1.49], [7.99, 1.51], [5.01, 4.49], [4.99, 1.51]],
            true_source_xy=(5.0, 3.0),
            rt60_target_s=0.30,
            snr_db=20.0,
            signal_type="broadband_noise",
        ),
        "demo_5_real_playback": DemoScenarioConfig(
            scenario_id="demo_5_real_playback",
            title="Demo 5: Real Multichannel Playback (LOCATA)",
            description="Genuine recorded multichannel acoustic corpus with real talker.",
            expected_outcome="Sequential causal localization tracks speaker across audio frames.",
            provenance=DataProvenance.REAL_RECORDED_DATASET,
            recommended_methods=["M2_Raw_SRP", "M12_Confidence_Hybrid"],
            room_dimensions_m=[7.1, 9.8, 3.0],
            mic_positions_2d=[[2.55, 4.9], [4.55, 4.9], [3.55, 5.2], [3.55, 4.6]],
            true_source_xy=(4.75, 6.9),
            rt60_target_s=0.55,
            snr_db=18.0,
            signal_type="real_speech",
        ),
    }

    @classmethod
    def generate_scenario_audio(cls, scenario_id: str, fs: int = 16000) -> tuple[np.ndarray, DemoScenarioConfig]:
        """Synthesize genuine acoustic audio matching the scenario configuration."""
        cfg = cls.SCENARIOS.get(scenario_id, cls.SCENARIOS["demo_1_stable"])

        if cfg.scenario_id == "demo_5_real_playback":
            adapter = LOCATADatasetAdapter(fs=fs)
            audio, meta = adapter.load_scene("LOCATA_Live_Demo")
            sub_mics = adapter.native_mics_15[adapter.subindices]
            cfg.mic_positions_2d = sub_mics.tolist()
            cfg.true_source_xy = (meta.true_source_position_3d[0], meta.true_source_position_3d[1])
            return audio[adapter.subindices], cfg

        # Synthesize real signal
        t_len = 0.25
        t_samples = int(fs * t_len)
        t = np.linspace(0, t_len, t_samples)

        if cfg.signal_type == "voiced_speech":
            sig, _ = generate_adversarial_speech_utterance("voiced_speech", f0_hz=140.0, fs=fs, seed=6001)
        elif cfg.signal_type == "linear_chirp":
            sig = np.sin(2 * np.pi * (300 + 3500 * (t / t_len)) * t)
        elif cfg.signal_type == "male_speech":
            sig, _ = generate_adversarial_speech_utterance("voiced_speech", f0_hz=110.0, fs=fs, seed=6002)
        else:
            sig = np.random.randn(t_samples)

        src_pos_3d = [cfg.true_source_xy[0], cfg.true_source_xy[1], 1.20]
        mics_arr = np.array(cfg.mic_positions_2d, dtype=np.float64)

        raw_audio, _ = simulate_room_acoustics(
            cfg.room_dimensions_m,
            mics_arr,
            src_pos_3d,
            sig,
            cfg.rt60_target_s,
            fs=fs,
        )
        noisy_audio, _ = generate_exact_snr_noise(raw_audio, target_snr_db=cfg.snr_db, seed=6003)
        return noisy_audio, cfg
