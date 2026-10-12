"""
EchoLocate — Speaker Enrollment Database
========================================
Manages enrolled speaker profiles, incremental centroid updates,
and serialized profile storage without saving raw audio by default.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..encoders.base import SpeakerEncoder
from ..preprocessing.audio_processor import AudioProcessor
from ..types import SpeakerProfile

logger = logging.getLogger(__name__)


def compute_quality_weight(m: Dict[str, Any]) -> float:
    """Calculate quality weight for an enrollment sample based on SNR, speech ratio, and clipping."""
    snr = float(m.get("snr_db", 25.0))
    ratio = float(m.get("speech_ratio", 0.8))
    clipped = bool(m.get("is_clipped", False))
    w = max(0.3, min(1.0, (snr + 5.0) / 35.0)) * max(0.4, min(1.0, ratio / 0.7))
    if clipped:
        w *= 0.5
    return float(max(0.2, min(1.0, w)))


class SpeakerEnrollmentDB:
    """
    In-memory and file-backed speaker enrollment database.
    Stores unit-normalized embedding centroids per enrolled identity.
    """

    def __init__(self, db_dir: Optional[Union[str, Path]] = None):
        self.db_dir = Path(db_dir) if db_dir is not None else None
        self._profiles: Dict[str, SpeakerProfile] = {}

        if self.db_dir is not None and self.db_dir.exists():
            self.load()

    def enroll_embedding(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpeakerProfile:
        """
        Enroll a single speaker embedding vector.
        If the speaker is already enrolled, updates the centroid incrementally.
        """
        # Ensure L2-normalized float32
        norm = np.linalg.norm(embedding)
        if norm > 1e-9:
            emb = (embedding / norm).astype(np.float32)
        else:
            raise ValueError(f"Cannot enroll zero-norm embedding for speaker {speaker_id}")

        meta = metadata or {}
        q_weight = compute_quality_weight(meta)

        if speaker_id in self._profiles:
            existing = self._profiles[speaker_id]
            if existing.model_name != model_name:
                logger.warning(
                    "Model mismatch during enrollment for %s: existing is %s, new is %s.",
                    speaker_id,
                    existing.model_name,
                    model_name,
                )

            # Duplicate enrollment check (Phase 10)
            if existing.embeddings:
                sims = [float(np.dot(emb, prev_e)) for prev_e in existing.embeddings]
                max_sim = max(sims)
                if max_sim > 0.96:
                    logger.warning(
                        "Duplicate/near-identical sample for '%s' (similarity: %.3f). Adds minimal diversity.",
                        speaker_id,
                        max_sim,
                    )

            # Retain all enrollment embeddings (Phase 8)
            all_embs = list(existing.embeddings) if existing.embeddings else [existing.centroid]
            all_embs.append(emb)

            # Exact centroid math with quality weighting (Phase 7 & Phase 9)
            # centroid = normalize(sum(w_i * emb_i))
            weights = existing.metadata.get("sample_weights", [1.0] * (len(all_embs) - 1))
            weights = list(weights) + [q_weight]
            weighted_sum = np.sum([w * e for w, e in zip(weights, all_embs)], axis=0)
            new_centroid = (weighted_sum / np.linalg.norm(weighted_sum)).astype(np.float32)

            updated_meta = {
                **existing.metadata,
                **meta,
                "sample_weights": weights,
                "last_updated": time.time(),
            }

            profile = SpeakerProfile(
                speaker_id=speaker_id,
                centroid=new_centroid,
                sample_count=len(all_embs),
                embedding_dim=len(new_centroid),
                model_name=model_name,
                created_at=existing.created_at,
                metadata=updated_meta,
                embeddings=all_embs,
            )
        else:
            weights = [q_weight]
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                centroid=emb,
                sample_count=1,
                embedding_dim=len(emb),
                model_name=model_name,
                created_at=time.time(),
                metadata={**meta, "sample_weights": weights},
                embeddings=[emb],
            )

        self._profiles[speaker_id] = profile
        logger.info(
            "Enrolled speaker '%s' (samples: %d, dim: %d, model: %s)",
            speaker_id,
            profile.sample_count,
            profile.embedding_dim,
            profile.model_name,
        )
        return profile

    def enroll_audio(
        self,
        speaker_id: str,
        audio: np.ndarray,
        encoder: SpeakerEncoder,
        sample_rate: int = 16000,
        preprocessor: Optional[AudioProcessor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpeakerProfile:
        """
        Preprocess audio, extract embedding via encoder, and update enrolled centroid.
        """
        if preprocessor is not None:
            seg = preprocessor.process(audio, sample_rate=sample_rate)
            proc_audio = seg.samples
            sr = seg.sample_rate
        else:
            proc_audio = audio
            sr = sample_rate

        emb = encoder.embed(proc_audio, sample_rate=sr)
        return self.enroll_embedding(
            speaker_id=speaker_id,
            embedding=emb,
            model_name=encoder.model_name(),
            metadata=metadata,
        )

    def enroll_multiple_samples(
        self,
        speaker_id: str,
        embeddings: List[np.ndarray],
        model_name: str,
        weights: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpeakerProfile:
        """
        Compute batch centroid over multiple enrollment samples at once.
        Supports quality-weighted centroid (Phase 7 & Phase 9).
        """
        if not embeddings:
            raise ValueError("No embeddings provided for enrollment")

        stacked = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embeddings], axis=0)
        if weights is not None and len(weights) == len(embeddings):
            w_arr = np.array(weights, dtype=np.float32).reshape(-1, 1)
            weighted = np.sum(stacked * w_arr, axis=0)
            centroid = (weighted / (np.linalg.norm(weighted) + 1e-9)).astype(np.float32)
        else:
            centroid = np.mean(stacked, axis=0)
            centroid = (centroid / (np.linalg.norm(centroid) + 1e-9)).astype(np.float32)

        meta = dict(metadata or {})
        if weights is not None:
            meta["sample_weights"] = list(weights)

        profile = SpeakerProfile(
            speaker_id=speaker_id,
            centroid=centroid,
            sample_count=len(embeddings),
            embedding_dim=len(centroid),
            model_name=model_name,
            created_at=time.time(),
            metadata=meta,
            embeddings=list(stacked),
        )
        self._profiles[speaker_id] = profile
        return profile

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        return self._profiles.get(speaker_id)

    def list_speakers(self) -> List[str]:
        return sorted(list(self._profiles.keys()))

    def remove_speaker(self, speaker_id: str) -> bool:
        if speaker_id in self._profiles:
            del self._profiles[speaker_id]
            return True
        return False

    def get_all_centroids(self) -> Tuple[List[str], np.ndarray]:
        """
        Returns list of speaker IDs and 2D matrix of centroids shape (K, D).
        """
        speaker_ids = self.list_speakers()
        if not speaker_ids:
            return [], np.zeros((0, 0), dtype=np.float32)

        centroids = np.stack([self._profiles[s].centroid for s in speaker_ids], axis=0)
        return speaker_ids, centroids

    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """
        Returns mapping of speaker ID to list of individual enrollment embeddings.
        """
        return {s: self._profiles[s].embeddings for s in self.list_speakers() if self._profiles[s].embeddings}

    def save(self, db_dir: Optional[Union[str, Path]] = None) -> None:
        target_dir = Path(db_dir) if db_dir is not None else self.db_dir
        if target_dir is None:
            raise ValueError("Target directory not specified for saving enrollment database")

        target_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {"version": "1.0.0", "speakers": {}}

        for s_id, prof in self._profiles.items():
            emb_file = f"{s_id}_centroid.npy"
            np.save(target_dir / emb_file, prof.centroid)
            manifest["speakers"][s_id] = {
                "speaker_id": prof.speaker_id,
                "centroid_file": emb_file,
                "sample_count": prof.sample_count,
                "embedding_dim": prof.embedding_dim,
                "model_name": prof.model_name,
                "created_at": prof.created_at,
                "metadata": prof.metadata,
            }

            # Also maintain individual speaker directory: <target_dir>/<speaker_id>/
            spk_dir = target_dir / s_id
            spk_dir.mkdir(parents=True, exist_ok=True)
            np.save(spk_dir / "centroid.npy", prof.centroid)
            if prof.embeddings:
                np.save(spk_dir / "embeddings.npy", np.stack(prof.embeddings, axis=0))

            spk_profile = {
                "speaker_id": prof.speaker_id,
                "display_name": prof.metadata.get("display_name", prof.speaker_id),
                "model_name": prof.model_name,
                "embedding_dim": prof.embedding_dim,
                "sample_count": prof.sample_count,
                "created_at": prof.created_at,
                "last_updated": prof.metadata.get("last_updated", time.time()),
                "metadata": prof.metadata,
            }
            with open(spk_dir / "profile.json", "w", encoding="utf-8") as pf:
                json.dump(spk_profile, pf, indent=2)

        with open(target_dir / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved %d speaker profiles to %s", len(self._profiles), target_dir)

    def load(self, db_dir: Optional[Union[str, Path]] = None) -> None:
        source_dir = Path(db_dir) if db_dir is not None else self.db_dir
        if source_dir is None or not source_dir.exists():
            return

        # 1. Load from legacy / top-level profiles.json
        manifest_file = source_dir / "profiles.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            for s_id, data in manifest.get("speakers", {}).items():
                emb_file = source_dir / data.get("centroid_file", f"{s_id}_centroid.npy")
                if not emb_file.exists():
                    emb_file = source_dir / s_id / "centroid.npy"
                if emb_file.exists():
                    centroid = np.load(emb_file)
                    # Load individual embeddings if present
                    embeddings = []
                    emb_arr_file = source_dir / s_id / "embeddings.npy"
                    if emb_arr_file.exists():
                        try:
                            embeddings = [e for e in np.load(emb_arr_file)]
                        except Exception:
                            pass
                    self._profiles[s_id] = SpeakerProfile(
                        speaker_id=s_id,
                        centroid=centroid,
                        sample_count=data.get("sample_count", 1),
                        embedding_dim=data.get("embedding_dim", len(centroid)),
                        model_name=data.get("model_name", "unknown"),
                        created_at=data.get("created_at", time.time()),
                        metadata=data.get("metadata", {}),
                        embeddings=embeddings,
                    )
            logger.info("Loaded %d profiles from manifest %s", len(self._profiles), manifest_file)
            return

        # 2. Directory-based profile loading
        for spk_dir in source_dir.iterdir():
            if spk_dir.is_dir() and (spk_dir / "profile.json").exists():
                with open(spk_dir / "profile.json", "r", encoding="utf-8") as pf:
                    data = json.load(pf)
                c_file = spk_dir / "centroid.npy"
                if c_file.exists():
                    centroid = np.load(c_file)
                    s_id = data.get("speaker_id", spk_dir.name)
                    embeddings = []
                    emb_arr_file = spk_dir / "embeddings.npy"
                    if emb_arr_file.exists():
                        try:
                            embeddings = [e for e in np.load(emb_arr_file)]
                        except Exception:
                            pass
                    self._profiles[s_id] = SpeakerProfile(
                        speaker_id=s_id,
                        centroid=centroid,
                        sample_count=data.get("sample_count", 1),
                        embedding_dim=data.get("embedding_dim", len(centroid)),
                        model_name=data.get("model_name", "unknown"),
                        created_at=data.get("created_at", time.time()),
                        metadata=data.get("metadata", {}),
                        embeddings=embeddings,
                    )

        # 2. Check for individual directory profiles: source_dir/*/profile.json
        for prof_path in source_dir.glob("*/profile.json"):
            s_id = prof_path.parent.name
            if s_id in self._profiles:
                continue
            try:
                with open(prof_path, "r", encoding="utf-8") as pf:
                    pdata = json.load(pf)
                cent_file = prof_path.parent / "centroid.npy"
                if not cent_file.exists():
                    cent_file = source_dir / f"{s_id}_centroid.npy"
                if cent_file.exists():
                    centroid = np.load(cent_file)
                    embeddings = []
                    emb_file = prof_path.parent / "embeddings.npy"
                    if emb_file.exists():
                        try:
                            embeddings = [e for e in np.load(emb_file)]
                        except Exception:
                            pass
                    self._profiles[s_id] = SpeakerProfile(
                        speaker_id=s_id,
                        centroid=centroid,
                        sample_count=pdata.get("sample_count", 1),
                        embedding_dim=pdata.get("embedding_dim", len(centroid)),
                        model_name=pdata.get("model_name", "unknown"),
                        created_at=pdata.get("created_at", time.time()),
                        metadata=pdata.get("metadata", {}),
                        embeddings=embeddings,
                    )
            except Exception as e:
                logger.warning("Failed to load speaker profile from %s: %s", prof_path, e)

        logger.info("Loaded %d speaker profiles from %s", len(self._profiles), source_dir)

    def delete_speaker(self, speaker_id: str) -> bool:
        """Remove speaker profile from memory and delete all associated local files and recordings."""
        if speaker_id not in self._profiles:
            return False

        del self._profiles[speaker_id]
        if self.db_dir is not None and self.db_dir.exists():
            flat_file = self.db_dir / f"{speaker_id}_centroid.npy"
            if flat_file.exists():
                flat_file.unlink(missing_ok=True)

            spk_dir = self.db_dir / speaker_id
            if spk_dir.exists() and spk_dir.is_dir():
                import shutil
                shutil.rmtree(spk_dir, ignore_errors=True)

            self.save()
        return True

    def __contains__(self, speaker_id: str) -> bool:
        return speaker_id in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self):
        return iter(self._profiles)

    def get_enrolled_speakers(self) -> List[str]:
        return self.list_speakers()
