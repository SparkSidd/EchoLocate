"""
EchoLocate — Tests: End-to-End Speaker Identifier & Benchmark Suite
===================================================================
Validates end-to-end pipeline execution, silence handling, multichannel policies,
and experiment runner integrity.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.speaker_attribution.backends.multichannel.multichannel_research import MultichannelSpeakerProcessor
from src.speaker_attribution.encoders.factory import create_encoder
from src.speaker_attribution.experiments.ablation_runner import Experiment7AblationRunner
from src.speaker_attribution.experiments.benchmark_runner import SpeakerModelBenchmarkRunner
from src.speaker_attribution.experiments.synthetic_generator import AcousticSpeakerGenerator
from src.speaker_attribution.identifier import SpeakerIdentifier
from src.speaker_attribution.types import AttributionStatus


class TestEndToEndSpeakerIdentifier:
    @pytest.fixture
    def setup_pipeline(self):
        gen = AcousticSpeakerGenerator(sample_rate=16000, seed=123)
        encoder = create_encoder("mock")
        ident = SpeakerIdentifier(
            encoder=encoder,
            unknown_threshold=0.60,
            enable_temporal_smoothing=False,
        )

        # Enroll 2 speakers
        a_siddharth = gen.generate_utterance("Siddharth", duration_sec=2.0, utterance_seed=1)
        a_alice = gen.generate_utterance("Alice", duration_sec=2.0, utterance_seed=2)

        ident.enroll("Siddharth", a_siddharth)
        ident.enroll("Alice", a_alice)

        return ident, gen

    def test_identify_known_and_impostor(self, setup_pipeline):
        ident, gen = setup_pipeline

        # 1. Test known speaker Siddharth
        test_sid = gen.generate_utterance("Siddharth", duration_sec=2.0, utterance_seed=3)
        res_sid = ident.identify(test_sid)
        assert res_sid.status == AttributionStatus.KNOWN
        assert res_sid.speaker_id == "Siddharth"
        assert res_sid.confidence >= 0.50

        # 2. Test unseen impostor (Eve)
        test_eve = gen.generate_utterance("Impostor_Eve", duration_sec=2.0, utterance_seed=4)
        res_eve = ident.identify(test_eve)
        # MUST reject as UNKNOWN
        assert res_eve.status == AttributionStatus.UNKNOWN
        assert res_eve.speaker_id == "UNKNOWN"

    def test_silence_handling(self, setup_pipeline):
        ident, _ = setup_pipeline
        silence = np.zeros(16000, dtype=np.float32)
        res = ident.identify(silence)
        assert res.status == AttributionStatus.SILENCE
        assert res.speaker_id == "UNKNOWN"


class TestMultichannelResearch:
    def test_multichannel_processor(self):
        gen = AcousticSpeakerGenerator(sample_rate=16000, seed=99)
        encoder = create_encoder("mock")
        processor = MultichannelSpeakerProcessor(encoder=encoder)

        sig = gen.generate_utterance("Siddharth", duration_sec=1.5)
        # 3 channels with channel 1 having best SNR
        ch0 = sig + 0.1 * np.random.randn(len(sig)).astype(np.float32)
        ch1 = sig + 0.001 * np.random.randn(len(sig)).astype(np.float32)
        ch2 = sig + 0.2 * np.random.randn(len(sig)).astype(np.float32)
        multi = np.stack([ch0, ch1, ch2], axis=0)

        target_cent = encoder.embed(sig)
        comp = processor.compare_policies(multi, target_cent)

        assert comp.single_channel_sim > 0.0
        assert comp.spatial_fusion_sim > 0.0
        assert len(comp.per_channel_snrs) == 3
        assert comp.best_policy in ["single_channel", "channel_average", "best_channel_snr", "spatial_fusion"]


class TestExperimentRunners:
    def test_benchmark_runner_quick(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SpeakerModelBenchmarkRunner(output_dir=tmp_dir)
            # Run fast benchmark on mock encoder
            results = runner.run_benchmark(models=["mock"], num_trials_per_condition=2)
            assert "mock" in results
            assert (Path(tmp_dir) / "mock.json").exists()
            assert (Path(tmp_dir) / "benchmark_summary.csv").exists()

    def test_ablation_runner_quick(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ab_runner = Experiment7AblationRunner(output_dir=tmp_dir, model_name="mock")
            ablations = ab_runner.run_all_ablations(num_trials=2)
            assert "E7.1" in ablations
            assert "E7.3" in ablations
            assert "E7.5" in ablations
            assert (Path(tmp_dir) / "ablation_results.json").exists()
