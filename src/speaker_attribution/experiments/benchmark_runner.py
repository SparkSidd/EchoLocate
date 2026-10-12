"""
EchoLocate — Model Benchmark Experiment Runner
==============================================
Evaluates CAM++, ERes2NetV2, and ECAPA-TDNN under an identical protocol.
Measures closed-set accuracy, EER, UNKNOWN rejection, noise, reverb, and short-utterance robustness.
Outputs machine-readable JSON and CSV results into results/speaker_attribution/.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..encoders.factory import create_encoder
from ..enrollment.database import SpeakerEnrollmentDB
from ..matching.calibration import ThresholdCalibrator
from ..matching.matcher import CosineSpeakerMatcher
from ..types import AttributionStatus
from .synthetic_generator import AcousticSpeakerGenerator

logger = logging.getLogger(__name__)


class SpeakerModelBenchmarkRunner:
    """
    Standardized benchmark runner for multi-architecture speaker verification.
    """

    def __init__(
        self,
        output_dir: str | Path = "results/speaker_attribution",
        sample_rate: int = 16000,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.gen = AcousticSpeakerGenerator(sample_rate=sample_rate, seed=seed)

        # Dataset partition
        self.enrolled_speakers = ["Siddharth", "Alice", "Bob", "Charlie"]
        self.impostor_speakers = ["Impostor_Eve", "Impostor_Mallory"]

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        num_trials_per_condition: int = 10,
    ) -> Dict[str, Any]:
        """
        Run master benchmark over candidate models.
        """
        candidate_models = models or ["campp", "eres2netv2", "ecapa"]
        benchmark_results: Dict[str, Any] = {}

        logger.info("================================================================================")
        logger.info("Starting Speaker Recognition Model Benchmark across: %s", candidate_models)
        logger.info("================================================================================")

        for model_name in candidate_models:
            logger.info("--> Evaluating Model: %s", model_name.upper())
            res = self._evaluate_single_model(model_name, num_trials=num_trials_per_condition)
            benchmark_results[model_name] = res

            # Save individual model JSON
            json_file = self.output_dir / f"{model_name}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            logger.info("Saved %s results to %s", model_name, json_file)

        # Generate comparative CSV
        csv_file = self.output_dir / "benchmark_summary.csv"
        self._export_summary_csv(benchmark_results, csv_file)
        logger.info("Saved benchmark summary table to %s", csv_file)

        return benchmark_results

    def _evaluate_single_model(self, model_name: str, num_trials: int = 10) -> Dict[str, Any]:
        encoder = create_encoder(model_name)
        db = SpeakerEnrollmentDB()

        # 1. Measure model parameter count & latency
        num_params = sum(p.numel() for p in getattr(encoder, "_model", torch.nn.Module()).parameters()) if hasattr(encoder, "_model") else 0
        sample_audio = self.gen.generate_utterance("Siddharth", duration_sec=2.0)

        t0 = time.perf_counter()
        for _ in range(5):
            _ = encoder.embed(sample_audio, sample_rate=self.sample_rate)
        t_embed = (time.perf_counter() - t0) / 5.0
        rtf = t_embed / 2.0  # Real-Time Factor (processing time / audio duration)

        # 2. Multi-sample Enrollment (3 samples per enrolled speaker)
        for s_id in self.enrolled_speakers:
            for s_idx in range(3):
                audio = self.gen.generate_utterance(s_id, duration_sec=2.5, utterance_seed=100 + s_idx)
                db.enroll_audio(s_id, audio, encoder, sample_rate=self.sample_rate)

        enrolled_ids, centroids = db.get_all_centroids()

        # 3. Validation set for Threshold Calibration (EER, FAR, FRR)
        val_scores: List[float] = []
        val_labels: List[bool] = []

        for s_id in self.enrolled_speakers:
            # Genuine trials
            for trial in range(num_trials):
                test_audio = self.gen.generate_utterance(s_id, duration_sec=2.0, utterance_seed=200 + trial)
                emb = encoder.embed(test_audio, sample_rate=self.sample_rate)
                target_centroid = db.get_profile(s_id).centroid
                val_scores.append(float(np.dot(emb, target_centroid)))
                val_labels.append(True)

            # Impostor trials (cross-speaker + unseen impostor)
            for imp_id in self.impostor_speakers:
                for trial in range(num_trials // 2):
                    imp_audio = self.gen.generate_utterance(imp_id, duration_sec=2.0, utterance_seed=300 + trial)
                    emb = encoder.embed(imp_audio, sample_rate=self.sample_rate)
                    target_centroid = db.get_profile(s_id).centroid
                    val_scores.append(float(np.dot(emb, target_centroid)))
                    val_labels.append(False)

        calibrator = ThresholdCalibrator()
        calib_report = calibrator.calibrate(val_scores, val_labels)
        opt_th = calib_report.optimal_threshold
        matcher = CosineSpeakerMatcher(default_threshold=opt_th)

        # 4. Closed-Set Identification Test (Rank-1 Accuracy)
        correct_top1 = 0
        total_closed = 0
        for s_id in self.enrolled_speakers:
            for trial in range(num_trials):
                test_audio = self.gen.generate_utterance(s_id, duration_sec=2.0, utterance_seed=500 + trial)
                emb = encoder.embed(test_audio, sample_rate=self.sample_rate)
                res = matcher.match(emb, enrolled_ids, centroids, model_name=model_name)
                top_id = res.alternatives[0].speaker_id if res.alternatives else ""
                if top_id == s_id:
                    correct_top1 += 1
                total_closed += 1
        closed_set_acc = float(correct_top1 / total_closed) if total_closed > 0 else 0.0

        # 5. UNKNOWN Speaker Rejection Test
        unknown_rejected = 0
        total_unknown = 0
        for imp_id in self.impostor_speakers:
            for trial in range(num_trials):
                imp_audio = self.gen.generate_utterance(imp_id, duration_sec=2.0, utterance_seed=700 + trial)
                emb = encoder.embed(imp_audio, sample_rate=self.sample_rate)
                res = matcher.match(emb, enrolled_ids, centroids, model_name=model_name)
                if res.status == AttributionStatus.UNKNOWN or res.speaker_id == "UNKNOWN":
                    unknown_rejected += 1
                total_unknown += 1
        unknown_rejection_rate = float(unknown_rejected / total_unknown) if total_unknown > 0 else 0.0

        # 6. Robustness Study: Duration Degradation (0.5s, 1.0s, 2.0s, 3.0s)
        durations = [0.5, 1.0, 2.0, 3.0]
        dur_acc = {}
        for dur in durations:
            corr = 0
            tot = 0
            for s_id in self.enrolled_speakers:
                for trial in range(5):
                    audio = self.gen.generate_utterance(s_id, duration_sec=dur, utterance_seed=800 + trial)
                    emb = encoder.embed(audio, sample_rate=self.sample_rate)
                    res = matcher.match(emb, enrolled_ids, centroids, model_name=model_name)
                    top_id = res.alternatives[0].speaker_id if res.alternatives else ""
                    if top_id == s_id:
                        corr += 1
                    tot += 1
            dur_acc[f"{dur}s"] = float(corr / tot)

        # 7. Robustness Study: Noise Degradation (Clean, 20dB, 10dB, 0dB SNR)
        snrs = [None, 20.0, 10.0, 0.0]
        snr_acc = {}
        for snr in snrs:
            label = "clean" if snr is None else f"{int(snr)}dB"
            corr = 0
            tot = 0
            for s_id in self.enrolled_speakers:
                for trial in range(5):
                    audio = self.gen.generate_utterance(s_id, duration_sec=2.0, snr_db=snr, utterance_seed=900 + trial)
                    emb = encoder.embed(audio, sample_rate=self.sample_rate)
                    res = matcher.match(emb, enrolled_ids, centroids, model_name=model_name)
                    top_id = res.alternatives[0].speaker_id if res.alternatives else ""
                    if top_id == s_id:
                        corr += 1
                    tot += 1
            snr_acc[label] = float(corr / tot)

        # 8. Robustness Study: Reverberation Degradation (Anechoic, 0.2s, 0.4s, 0.6s)
        rt60s = [0.0, 0.2, 0.4, 0.6]
        reverb_acc = {}
        for rt in rt60s:
            label = f"RT60_{rt}s"
            corr = 0
            tot = 0
            for s_id in self.enrolled_speakers:
                for trial in range(5):
                    audio = self.gen.generate_utterance(s_id, duration_sec=2.0, rt60_sec=rt, utterance_seed=1000 + trial)
                    emb = encoder.embed(audio, sample_rate=self.sample_rate)
                    res = matcher.match(emb, enrolled_ids, centroids, model_name=model_name)
                    top_id = res.alternatives[0].speaker_id if res.alternatives else ""
                    if top_id == s_id:
                        corr += 1
                    tot += 1
            reverb_acc[label] = float(corr / tot)

        return {
            "model_name": model_name,
            "embedding_dimension": encoder.embedding_dimension(),
            "parameter_count": num_params,
            "inference_latency_ms": round(t_embed * 1000.0, 2),
            "real_time_factor": round(rtf, 4),
            "calibrated_threshold": round(opt_th, 4),
            "eer_percent": round(calib_report.eer * 100.0, 2),
            "auc_roc": round(calib_report.auc_roc, 4),
            "closed_set_accuracy_percent": round(closed_set_acc * 100.0, 2),
            "unknown_rejection_rate_percent": round(unknown_rejection_rate * 100.0, 2),
            "duration_robustness": dur_acc,
            "noise_robustness": snr_acc,
            "reverb_robustness": reverb_acc,
        }

    def _export_summary_csv(self, results: Dict[str, Any], filepath: Path) -> None:
        headers = [
            "Model",
            "Embedding Dim",
            "Params",
            "Latency (ms)",
            "RTF",
            "Threshold",
            "EER (%)",
            "AUC ROC",
            "Closed-Set Acc (%)",
            "UNKNOWN Rejection (%)",
            "Acc @ 0.5s",
            "Acc @ 1.0s",
            "Acc @ 10dB SNR",
            "Acc @ RT60 0.4s",
        ]

        rows = []
        for model_name, data in results.items():
            rows.append([
                data["model_name"],
                data["embedding_dimension"],
                data["parameter_count"],
                data["inference_latency_ms"],
                data["real_time_factor"],
                data["calibrated_threshold"],
                data["eer_percent"],
                data["auc_roc"],
                data["closed_set_accuracy_percent"],
                data["unknown_rejection_rate_percent"],
                data["duration_robustness"].get("0.5s", "-"),
                data["duration_robustness"].get("1.0s", "-"),
                data["noise_robustness"].get("10dB", "-"),
                data["reverb_robustness"].get("RT60_0.4s", "-"),
            ])

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
