"""
EchoLocate — Human Speaker Demonstration Session Manager
=========================================================
Orchestrates enrollment, real-time identification, unknown rejection,
and blind testing with full metrics aggregation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..encoders.base import SpeakerEncoder
from ..encoders.factory import create_encoder
from ..enrollment.database import SpeakerEnrollmentDB, compute_quality_weight
from ..identifier import SpeakerIdentifier
from ..matching.matcher import CosineSpeakerMatcher
from ..temporal.tracker import TemporalSpeakerTracker
from ..types import AttributionStatus, SpeakerAttributionResult, SpeakerProfile
from .quality_checker import EnrollmentQualityChecker, QualityCheckResult

logger = logging.getLogger(__name__)


@dataclass
class BlindTestCase:
    """Individual blind evaluation test result."""

    test_id: int
    true_speaker: str
    predicted_speaker: str
    decision_status: str
    is_correct: bool
    similarity: float
    confidence: float
    threshold: float
    margin: float
    condition: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvaluationSummary:
    """Aggregate numerical metrics from a real human evaluation session."""

    enrolled_speakers_count: int
    total_test_utterances: int
    known_tests_count: int
    known_correct_count: int
    known_accuracy_percent: float
    unknown_tests_count: int
    unknown_correct_count: int
    unknown_rejection_percent: float
    overall_accuracy_percent: float
    mean_latency_ms: float
    condition_breakdown: Dict[str, Dict[str, Any]]
    test_cases: List[Dict[str, Any]]

    def format_report(self) -> str:
        """Format a clean textual report without verbose fluff."""
        return (
            "============================================================\n"
            "         REAL SPEAKER ATTRIBUTION EVALUATION REPORT         \n"
            "============================================================\n"
            f"Enrolled speakers:       {self.enrolled_speakers_count}\n"
            f"Total test utterances:   {self.total_test_utterances}\n\n"
            f"Known Speaker Tests:     {self.known_tests_count}\n"
            f"  Correct:               {self.known_correct_count}\n"
            f"  Incorrect:             {self.known_tests_count - self.known_correct_count}\n"
            f"  Known Accuracy:        {self.known_accuracy_percent:.1f}%\n\n"
            f"Unknown Impostor Tests:  {self.unknown_tests_count}\n"
            f"  Correct UNKNOWN:       {self.unknown_correct_count}\n"
            f"  False Acceptance:      {self.unknown_tests_count - self.unknown_correct_count}\n"
            f"  UNKNOWN Rejection:     {self.unknown_rejection_percent:.1f}%\n\n"
            f"Overall Decision Acc:    {self.overall_accuracy_percent:.1f}%\n"
            f"Mean Latency:            {self.mean_latency_ms:.2f} ms\n"
            "============================================================"
        )


class DemonstrationSession:
    """
    Manages state, profiles, and tests for live real human speaker attribution.
    """

    def __init__(
        self,
        db_dir: Union[str, Path] = "demo_data/speakers",
        model_name: str = "campp",
        default_threshold: float = 0.42,
        min_margin: float = 0.035,
    ):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.encoder: SpeakerEncoder = create_encoder(model_name)
        self.db = SpeakerEnrollmentDB(db_dir=self.db_dir)
        self.matcher = CosineSpeakerMatcher(
            default_threshold=default_threshold,
            min_margin=min_margin,
            confidence_scaling=10.0,
        )
        self.identifier = SpeakerIdentifier(
            encoder=self.encoder,
            enrollment_db=self.db,
            matcher=self.matcher,
            enable_temporal_smoothing=False,
        )
        self.quality_checker = EnrollmentQualityChecker(sample_rate=16000)
        self.temporal_tracker = TemporalSpeakerTracker(
            window_size=10,
            ema_alpha=0.35,
            switch_confirmation_frames=3,
        )

        self.blind_tests: List[BlindTestCase] = []
        # Ensure enrolled centroids always match active encoder dimensionality at startup
        self._sync_enrolled_embeddings()

    def _sync_enrolled_embeddings(self, force: bool = False) -> None:
        """
        Verify that all enrolled speaker centroids match the active encoder's
        embedding dimension. If any mismatch is detected (e.g. after model switch
        or server restart), dynamically recompute centroids from saved WAV recordings.
        """
        import soundfile as sf
        target_dim = self.encoder.embedding_dimension()
        need_sync = force
        if not need_sync:
            enrolled_spks = self.db.get_enrolled_speakers()
            if not enrolled_spks:
                need_sync = True
            for s_id in enrolled_spks:
                prof = self.db.get_profile(s_id)
                if prof and (prof.embedding_dim != target_dim or prof.model_name != self.model_name or len(prof.embeddings) == 0):
                    need_sync = True
                    break

        if not need_sync:
            return

        logger.info(
            "Synchronizing enrolled speaker centroids to active model '%s' (dim: %d)",
            self.model_name,
            target_dim,
        )
        spk_ids = set(self.db.get_enrolled_speakers())
        if self.db_dir.exists():
            for spk_d in self.db_dir.iterdir():
                if spk_d.is_dir() and (spk_d / "recordings").exists():
                    spk_ids.add(spk_d.name)

        for s_id in sorted(list(spk_ids)):
            prof = self.db.get_profile(s_id)
            rec_dir = self.db_dir / s_id / "recordings"
            wav_files = sorted(list(rec_dir.glob("*.wav"))) if rec_dir.exists() else []
            if not wav_files:
                continue

            embs = []
            weights = []
            for wf in wav_files:
                try:
                    data, sr = sf.read(str(wf), dtype="float32")
                    emb = self.encoder.embed(data, sample_rate=sr)
                    embs.append(emb)
                    qc = self.quality_checker.check(data, sample_rate=sr)
                    weights.append(compute_quality_weight(asdict(qc)))
                except Exception as e:
                    logger.warning("Failed to re-embed %s with %s: %s", wf, self.model_name, e)

            if embs:
                self.db.enroll_multiple_samples(
                    speaker_id=s_id,
                    embeddings=embs,
                    weights=weights,
                    model_name=self.model_name,
                    metadata=prof.metadata if prof else {},
                )
        self.db.save()

    def enroll_sample(
        self,
        speaker_id: str,
        display_name: str,
        audio: np.ndarray,
        allow_override_quality: bool = False,
    ) -> Tuple[bool, QualityCheckResult, Optional[SpeakerProfile]]:
        """
        Run quality gates and enroll an utterance into a speaker profile.
        Saves accepted recording locally to <db_dir>/<speaker_id>/recordings/<index>.wav.
        """
        quality = self.quality_checker.check(audio)
        if not quality.passed and not allow_override_quality:
            return False, quality, None

        # 1. Save local WAV recording
        spk_dir = self.db_dir / speaker_id
        rec_dir = spk_dir / "recordings"
        rec_dir.mkdir(parents=True, exist_ok=True)
        existing_wavs = list(rec_dir.glob("*.wav"))
        sample_num = len(existing_wavs) + 1
        wav_filename = f"{sample_num:03d}.wav"
        wav_path = rec_dir / wav_filename

        rel_path = f"recordings/{wav_filename}"
        try:
            import soundfile as sf
            sf.write(str(wav_path), audio, 16000, subtype="PCM_16")
        except Exception as e:
            logger.warning("Failed to save local WAV file %s: %s", wav_path, e)

        # 2. Track recordings history in profile metadata
        existing_prof = self.db.get_profile(speaker_id)
        existing_recs = []
        if existing_prof and existing_prof.metadata:
            existing_recs = list(existing_prof.metadata.get("recordings", []))
        if rel_path not in existing_recs:
            existing_recs.append(rel_path)

        emb = self.encoder.embed(audio)
        profile = self.db.enroll_embedding(
            speaker_id=speaker_id,
            embedding=emb,
            model_name=self.model_name,
            metadata={
                "display_name": display_name,
                "recordings": existing_recs,
                "speech_duration_sec": quality.speech_duration_sec,
                "rms": quality.rms,
                "snr_db": quality.snr_db,
                "last_sample_file": rel_path,
                "last_updated": time.time(),
            },
        )
        self.db.save()
        return True, quality, profile

    def switch_model(self, model_name: str) -> None:
        """
        Switch active speaker encoder and dynamically re-compute centroids for all
        enrolled speakers using their preserved local WAV recordings.
        """
        self.model_name = model_name
        self.encoder = create_encoder(model_name)
        self.identifier.encoder = self.encoder
        self._sync_enrolled_embeddings(force=True)

    def delete_speaker(self, speaker_id: str) -> bool:
        """Remove speaker profile, centroids, and all saved recordings from disk."""
        return self.db.delete_speaker(speaker_id)

    def get_speaker_details(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details of an enrolled speaker profile including recordings."""
        prof = self.db.get_profile(speaker_id)
        if not prof:
            return None
        rec_dir = self.db_dir / speaker_id / "recordings"
        wav_list = sorted([f.name for f in rec_dir.glob("*.wav")]) if rec_dir.exists() else []
        return {
            "speaker_id": prof.speaker_id,
            "display_name": prof.metadata.get("display_name", prof.speaker_id) if prof.metadata else prof.speaker_id,
            "model_name": prof.model_name,
            "embedding_dim": prof.embedding_dim,
            "sample_count": prof.sample_count,
            "created_at": prof.created_at,
            "last_updated": prof.metadata.get("last_updated", prof.created_at) if prof.metadata else prof.created_at,
            "recordings": wav_list,
            "metadata": prof.metadata or {},
        }

    def identify_sample(
        self,
        audio: np.ndarray,
        use_temporal_tracking: bool = False,
    ) -> Dict[str, Any]:
        """
        Identify speaker from live or recorded audio with timing diagnostics.
        """
        t0 = time.perf_counter()
        quality = self.quality_checker.check(audio)

        self._sync_enrolled_embeddings()
        raw_result = self.identifier.identify(audio)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Margin computation
        margin = 0.0
        if len(raw_result.alternatives) > 1:
            margin = raw_result.alternatives[0].similarity - raw_result.alternatives[1].similarity

        # Temporal smoothing
        smoothed_speaker = raw_result.speaker_id
        stability = 1.0
        if use_temporal_tracking:
            smooth_res = self.temporal_tracker.update(raw_result)
            smoothed_speaker = smooth_res.speaker_id
            stability = smooth_res.temporal_stability

        # Display name lookup
        display_name = raw_result.speaker_id
        if raw_result.speaker_id in self.db:
            prof = self.db.get_profile(raw_result.speaker_id)
            if prof and prof.metadata and "display_name" in prof.metadata:
                display_name = prof.metadata["display_name"]

        return {
            "speaker_id": raw_result.speaker_id,
            "display_name": display_name,
            "decision": raw_result.status.value.upper(),
            "confidence": round(float(raw_result.confidence), 3),
            "similarity": round(float(raw_result.similarity), 3),
            "threshold": round(float(raw_result.threshold), 3),
            "margin": round(float(margin), 3),
            "latency_ms": round(latency_ms, 2),
            "quality": asdict(quality),
            "smoothed_speaker": smoothed_speaker,
            "temporal_stability": round(stability, 2),
            "alternatives": [
                {
                    "speaker_id": a.speaker_id,
                    "similarity": round(float(a.similarity), 3),
                    "rank": a.rank,
                }
                for a in raw_result.alternatives[:5]
            ],
        }

    def record_blind_test(
        self,
        true_speaker: str,
        audio: np.ndarray,
        condition: str = "normal",
    ) -> BlindTestCase:
        """
        Execute an identification test without revealing ground truth to the user.
        """
        res = self.identify_sample(audio)
        pred_spk = res["speaker_id"]
        status = res["decision"]

        # If true speaker is not enrolled, ground truth is UNKNOWN
        is_enrolled = true_speaker in self.db
        expected_label = true_speaker if is_enrolled else "UNKNOWN"

        is_correct = (pred_spk == expected_label)

        test_case = BlindTestCase(
            test_id=len(self.blind_tests) + 1,
            true_speaker=true_speaker,
            predicted_speaker=pred_spk,
            decision_status=status,
            is_correct=is_correct,
            similarity=res["similarity"],
            confidence=res["confidence"],
            threshold=res["threshold"],
            margin=res["margin"],
            condition=condition,
            latency_ms=res["latency_ms"],
        )
        self.blind_tests.append(test_case)
        return test_case

    def compute_summary(self) -> EvaluationSummary:
        """
        Compute aggregate session metrics over all blind test cases.
        """
        total = len(self.blind_tests)
        enrolled_count = len(self.db)

        if total == 0:
            return EvaluationSummary(
                enrolled_speakers_count=enrolled_count,
                total_test_utterances=0,
                known_tests_count=0,
                known_correct_count=0,
                known_accuracy_percent=0.0,
                unknown_tests_count=0,
                unknown_correct_count=0,
                unknown_rejection_percent=0.0,
                overall_accuracy_percent=0.0,
                mean_latency_ms=0.0,
                condition_breakdown={},
                test_cases=[],
            )

        known_tests = [t for t in self.blind_tests if t.true_speaker in self.db]
        unknown_tests = [t for t in self.blind_tests if t.true_speaker not in self.db]

        known_corr = sum(1 for t in known_tests if t.is_correct)
        unknown_corr = sum(1 for t in unknown_tests if t.is_correct)
        total_corr = sum(1 for t in self.blind_tests if t.is_correct)

        known_acc = (100.0 * known_corr / len(known_tests)) if known_tests else 100.0
        unk_rej = (100.0 * unknown_corr / len(unknown_tests)) if unknown_tests else 100.0
        overall_acc = 100.0 * total_corr / total
        mean_lat = float(np.mean([t.latency_ms for t in self.blind_tests]))

        # Condition breakdown
        cond_map: Dict[str, Dict[str, Any]] = {}
        for t in self.blind_tests:
            if t.condition not in cond_map:
                cond_map[t.condition] = {"total": 0, "correct": 0, "latencies": []}
            cond_map[t.condition]["total"] += 1
            if t.is_correct:
                cond_map[t.condition]["correct"] += 1
            cond_map[t.condition]["latencies"].append(t.latency_ms)

        condition_breakdown = {}
        for c, stats in cond_map.items():
            condition_breakdown[c] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy_percent": round(100.0 * stats["correct"] / stats["total"], 1),
                "mean_latency_ms": round(float(np.mean(stats["latencies"])), 2),
            }

        return EvaluationSummary(
            enrolled_speakers_count=enrolled_count,
            total_test_utterances=total,
            known_tests_count=len(known_tests),
            known_correct_count=known_corr,
            known_accuracy_percent=round(known_acc, 2),
            unknown_tests_count=len(unknown_tests),
            unknown_correct_count=unknown_corr,
            unknown_rejection_percent=round(unk_rej, 2),
            overall_accuracy_percent=round(overall_acc, 2),
            mean_latency_ms=round(mean_lat, 2),
            condition_breakdown=condition_breakdown,
            test_cases=[asdict(t) for t in self.blind_tests],
        )

    def save_summary_json(self, output_path: Union[str, Path]) -> Path:
        """Export session evaluation summary to machine-readable JSON."""
        summary = self.compute_summary()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        return out

    def evaluate_speaker_profiles(
        self,
        split_first_n_enroll: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute machine-readable genuine vs impostor verification matrix (Phase 11 & Phase 19).
        Computes genuine/impostor distributions, margin, EER, ROC, FAR, FRR.
        """
        import soundfile as sf

        trials = []
        enrolled_speakers = self.db.get_enrolled_speakers()
        if not enrolled_speakers:
            return {"error": "no_enrolled_speakers", "trials": []}

        # Collect audio files per speaker
        speaker_wavs: Dict[str, List[Path]] = {}
        for s_id in enrolled_speakers:
            rec_dir = self.db_dir / s_id / "recordings"
            wavs = sorted(list(rec_dir.glob("*.wav"))) if rec_dir.exists() else []
            if wavs:
                speaker_wavs[s_id] = wavs

        if len(speaker_wavs) == 0:
            return {"error": "no_recordings_found", "trials": []}

        # If split_first_n_enroll is given (e.g. 5), build temporary enrollment DB from first N samples
        if split_first_n_enroll is not None and split_first_n_enroll > 0:
            eval_db = SpeakerEnrollmentDB()
            for s_id, wavs in speaker_wavs.items():
                train_wavs = wavs[:split_first_n_enroll]
                train_embs = []
                train_weights = []
                for tw in train_wavs:
                    data, sr = sf.read(str(tw), dtype="float32")
                    emb = self.encoder.embed(data, sample_rate=sr)
                    train_embs.append(emb)
                    qc = self.quality_checker.check(data, sample_rate=sr)
                    train_weights.append(compute_quality_weight(asdict(qc)))
                if train_embs:
                    eval_db.enroll_multiple_samples(
                        speaker_id=s_id,
                        embeddings=train_embs,
                        weights=train_weights,
                        model_name=self.model_name,
                    )
            test_target_db = eval_db
            eval_test_wavs = {s_id: wavs[split_first_n_enroll:] for s_id, wavs in speaker_wavs.items()}
        else:
            test_target_db = self.db
            eval_test_wavs = speaker_wavs

        all_ids, centroids = test_target_db.get_all_centroids()
        all_embs = test_target_db.get_all_embeddings()

        genuine_scores = []
        impostor_scores = []

        trial_counter = 0
        for actual_spk, wavs in eval_test_wavs.items():
            for wf in wavs:
                trial_counter += 1
                try:
                    data, sr = sf.read(str(wf), dtype="float32")
                except Exception:
                    continue

                duration = float(len(data) / sr)
                qc = self.quality_checker.check(data, sample_rate=sr)
                t0 = time.perf_counter()
                emb = self.encoder.embed(data, sample_rate=sr)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                res = self.matcher.match(
                    query_embedding=emb,
                    enrolled_ids=all_ids,
                    enrolled_centroids=centroids,
                    enrolled_embeddings=all_embs,
                    model_name=self.model_name,
                )

                actual_spk_idx = all_ids.index(actual_spk) if actual_spk in all_ids else -1
                if actual_spk_idx >= 0:
                    c_sim = float(np.dot(centroids[actual_spk_idx], emb / (np.linalg.norm(emb) + 1e-9)))
                    spk_e = all_embs.get(actual_spk, [])
                    if spk_e:
                        s_sims = [float(np.dot(e / (np.linalg.norm(e) + 1e-9), emb / (np.linalg.norm(emb) + 1e-9))) for e in spk_e]
                        t1 = max(s_sims)
                        t3 = float(np.mean(sorted(s_sims, reverse=True)[:min(3, len(s_sims))]))
                        w_cent, w_top1, w_top3 = self.matcher.fusion_weights
                        gen_score = float((w_cent * c_sim) + (w_top1 * t1) + (w_top3 * t3))
                    else:
                        gen_score = c_sim

                    genuine_scores.append(gen_score)
                    trials.append({
                        "trial_id": trial_counter,
                        "speaker": actual_spk,
                        "enrolled_target": actual_spk,
                        "trial_file": wf.name,
                        "score": round(gen_score, 4),
                        "is_genuine": True,
                        "predicted": res.speaker_id,
                        "duration": round(duration, 2),
                        "snr": round(qc.snr_db, 1),
                        "speech_ratio": round(qc.speech_ratio, 2),
                        "microphone": "browser_pcm",
                        "condition": "same_speaker",
                        "latency_ms": round(latency_ms, 2),
                    })

                for other_idx, other_spk in enumerate(all_ids):
                    if other_spk == actual_spk:
                        continue
                    c_sim_imp = float(np.dot(centroids[other_idx], emb / (np.linalg.norm(emb) + 1e-9)))
                    spk_e_imp = all_embs.get(other_spk, [])
                    if spk_e_imp:
                        s_sims_imp = [float(np.dot(e / (np.linalg.norm(e) + 1e-9), emb / (np.linalg.norm(emb) + 1e-9))) for e in spk_e_imp]
                        t1_imp = max(s_sims_imp)
                        t3_imp = float(np.mean(sorted(s_sims_imp, reverse=True)[:min(3, len(s_sims_imp))]))
                        w_cent, w_top1, w_top3 = self.matcher.fusion_weights
                        imp_score = float((w_cent * c_sim_imp) + (w_top1 * t1_imp) + (w_top3 * t3_imp))
                    else:
                        imp_score = c_sim_imp

                    impostor_scores.append(imp_score)
                    trials.append({
                        "trial_id": trial_counter,
                        "speaker": actual_spk,
                        "enrolled_target": other_spk,
                        "trial_file": wf.name,
                        "score": round(imp_score, 4),
                        "is_genuine": False,
                        "predicted": res.speaker_id,
                        "duration": round(duration, 2),
                        "snr": round(qc.snr_db, 1),
                        "speech_ratio": round(qc.speech_ratio, 2),
                        "microphone": "browser_pcm",
                        "condition": "impostor_cross_speaker",
                        "latency_ms": round(latency_ms, 2),
                    })

        gen_mean = float(np.mean(genuine_scores)) if genuine_scores else 0.0
        gen_std = float(np.std(genuine_scores)) if genuine_scores else 0.0
        imp_mean = float(np.mean(impostor_scores)) if impostor_scores else 0.0
        imp_std = float(np.std(impostor_scores)) if impostor_scores else 0.0
        score_margin = gen_mean - imp_mean

        min_gen = float(np.min(genuine_scores)) if genuine_scores else 0.0
        overlap_count = sum(1 for s in impostor_scores if s >= min_gen)
        overlap_ratio = float(overlap_count / len(impostor_scores)) if impostor_scores else 0.0

        threshold_candidates = np.linspace(0.10, 0.90, 81)
        roc_curve = []
        best_diff = 1e9
        eer = 0.5
        optimal_threshold = 0.42

        for th in threshold_candidates:
            frr = float(np.mean([1.0 if s < th else 0.0 for s in genuine_scores])) if genuine_scores else 0.0
            far = float(np.mean([1.0 if s >= th else 0.0 for s in impostor_scores])) if impostor_scores else 0.0
            roc_curve.append({
                "threshold": round(float(th), 3),
                "far": round(far, 4),
                "frr": round(frr, 4),
                "accuracy": round(1.0 - 0.5 * (far + frr), 4),
            })
            diff = abs(far - frr)
            if diff < best_diff:
                best_diff = diff
                eer = 0.5 * (far + frr)
                optimal_threshold = float(th)

        current_th = self.matcher.default_threshold
        cur_frr = float(np.mean([1.0 if s < current_th else 0.0 for s in genuine_scores])) if genuine_scores else 0.0
        cur_far = float(np.mean([1.0 if s >= current_th else 0.0 for s in impostor_scores])) if impostor_scores else 0.0

        return {
            "model_name": self.model_name,
            "enrolled_speakers": enrolled_speakers,
            "total_genuine_trials": len(genuine_scores),
            "total_impostor_trials": len(impostor_scores),
            "genuine_mean": round(gen_mean, 4),
            "genuine_std": round(gen_std, 4),
            "impostor_mean": round(imp_mean, 4),
            "impostor_std": round(imp_std, 4),
            "separation_margin": round(score_margin, 4),
            "score_overlap_ratio": round(overlap_ratio, 4),
            "eer": round(eer, 4),
            "optimal_threshold": round(optimal_threshold, 3),
            "active_threshold": round(current_th, 3),
            "active_threshold_far": round(cur_far, 4),
            "active_threshold_frr": round(cur_frr, 4),
            "roc_sample": [pt for i, pt in enumerate(roc_curve) if i % 10 == 0],
            "trials": trials,
        }
