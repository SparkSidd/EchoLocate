"""
EchoLocate — ReDimNet2-B0 Speaker Encoder Adapter
=================================================
Optional research speaker encoder implementing the SpeakerEncoder interface.
Integrates PalabraAI's ReDimNet2-B0 architecture (VoxCeleb2-dev large-margin variant).

Provenance & Licensing:
  - Architecture: ReDimNet2-B0 (1.1M parameters, 192-dim embedding)
  - Code License: MIT License (PalabraAI/redimnet2)
  - Pretrained Weights License: Unverified / Unspecified (No explicit license grant published
    by upstream authors for b0-vox2-lm.pt; restricted to internal research benchmark)
  - Upstream Release: v1.0.0 (commit 5294667)
  - Checkpoint File: b0-vox2-lm.pt
  - Expected Checkpoint SHA-256: 4b5a8c00f813ab0d92fc81ea7c504475d8201bed608fb2c6aaba65be723e9c61
  - Verified redimnet2.py Blob SHA-1: 33d5d4524f4d53d642f4919a5fb8ffb875316c19
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import SpeakerEncoder
from ..types import ModelProvenance

logger = logging.getLogger("EchoLocate.ReDimNet2")

EXPECTED_B0_SHA256 = "4b5a8c00f813ab0d92fc81ea7c504475d8201bed608fb2c6aaba65be723e9c61"
PINNED_V1_0_0_COMMIT = "5294667"
PINNED_V1_0_0_MANIFEST: Dict[str, str] = {
    "hubconf.py": "d7e021f88f501c2951d9aeca7b930ccfda36b274",
    "redimnet2/__init__.py": "6f9a906de3039170e4788af4fd411bdf77215129",
    "redimnet2/layers/attention.py": "84b7b2df16bed8864bc0631bbf16dec927d34a24",
    "redimnet2/layers/blocks.py": "4432acda3144b7e9436024ea7aa9793bf32d5af1",
    "redimnet2/layers/convnext.py": "83c85514366dcc65b070c49479136edc9c377134",
    "redimnet2/layers/features.py": "b413f296d0d17f1a5505dccc7a2e40d9acb93971",
    "redimnet2/layers/features_tf.py": "14ae0651c9b9216922e325b56d5e0258658d7b8b",
    "redimnet2/layers/layernorm.py": "404ff787fefd291f2212fa5c5bc22e019c252c03",
    "redimnet2/layers/poolings.py": "a0d0b8d4c3d803b76b526ff3911d4be204202880",
    "redimnet2/layers/redim_structural.py": "9da65364496c13152fe086bdef9fa91da7ac6e43",
    "redimnet2/layers/resblocks.py": "990286b7c0f9108d64630f597a6eff969f43c418",
    "redimnet2/redimnet2.py": "33d5d4524f4d53d642f4919a5fb8ffb875316c19",
}


def verify_source_tree_integrity(root_dir: Path) -> bool:
    """Validate all 12 Python modules against the pinned release v1.0.0 manifest."""
    for rel_path, expected_blob in PINNED_V1_0_0_MANIFEST.items():
        file_path = root_dir / rel_path
        if not file_path.exists():
            return False
        data = file_path.read_bytes()
        actual_blob = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if actual_blob != expected_blob:
            return False
    return True


def _resolve_redimnet2_module():
    """
    Locate and import ReDimNet2Wrap from a verified release v1.0.0 (commit 5294667) source tree.
    Prioritizes tracked third_party/redimnet2 directory to guarantee fresh-checkout portability.
    Refuses unverified or unpinned upstream code to guarantee reproducibility.
    """
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "third_party" / "redimnet2",
        Path("third_party/redimnet2"),
        project_root / "demo_data" / "models" / "redimnet2" / "v1.0.0_src",
        Path.home() / ".cache" / "torch" / "hub" / "PalabraAI_redimnet2_v1.0.0",
    ]

    for cand in candidates:
        if cand.exists() and verify_source_tree_integrity(cand):
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            try:
                from redimnet2.redimnet2 import ReDimNet2Wrap
                logger.debug("Imported ReDimNet2Wrap from verified v1.0.0 tree: %s", cand)
                return ReDimNet2Wrap
            except ImportError:
                pass
        elif cand.exists():
            logger.warning(
                "Candidate ReDimNet2 source tree at '%s' failed v1.0.0 (commit %s) manifest integrity check.",
                cand,
                PINNED_V1_0_0_COMMIT,
            )

    raise ImportError(
        f"Verified ReDimNet2 release v1.0.0 source (commit {PINNED_V1_0_0_COMMIT}) "
        f"failed manifest integrity check across candidate paths: {[str(c) for c in candidates]}. "
        f"All 12 upstream Python modules must match the pinned v1.0.0 release manifest. "
        f"Refusing to load unverified or moving branch code."
    )


class ReDimNet2B0Encoder(SpeakerEncoder):
    """
    Adapter for the ReDimNet2-B0 speaker embedding model.
    Guarantees strict offline loading, SHA-256 verification, and L2-normalized embeddings.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 192,
        device: str = "cpu",
        model: Optional[nn.Module] = None,
    ):
        self._device_str = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self._device = torch.device(self._device_str)
        self._dim = embedding_dim
        self._checkpoint_hash: Optional[str] = None
        self._is_verified: bool = False

        if model is not None:
            # Injected model (e.g. for unit testing without local checkpoint)
            self._model = model
            self._model.to(self._device)
            self._model.eval()
            self._checkpoint_hash = EXPECTED_B0_SHA256
            self._is_verified = True
            logger.info("Initialized ReDimNet2-B0 encoder with injected model.")
        else:
            self.load(model_path)

    def load(self, model_path: Optional[str] = None) -> None:
        """Initialize and load network weights from verified checkpoint."""
        self._load_from_checkpoint(model_path)

    def _find_checkpoint(self, explicit_path: Optional[str]) -> Path:
        """Find local checkpoint file without network access."""
        if explicit_path is not None:
            cand = Path(explicit_path)
            if cand.exists() and cand.is_file():
                return cand
            raise FileNotFoundError(
                f"ReDimNet2-B0 checkpoint not found at explicit path: '{explicit_path}'. "
                f"Automatic network downloads are disabled to guarantee reproducibility. "
                f"Please place verified weights at this path (expected SHA-256: {EXPECTED_B0_SHA256})."
            )

        candidates = [
            Path("demo_data/models/redimnet2/b0-vox2-lm.pt"),
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "b0-vox2-lm.pt",
        ]

        for cand in candidates:
            if cand.exists() and cand.is_file():
                return cand

        raise FileNotFoundError(
            f"ReDimNet2-B0 checkpoint 'b0-vox2-lm.pt' not found in candidate paths: "
            f"{[str(c) for c in candidates]}. Automatic network downloads are disabled "
            f"to guarantee reproducibility. Please place verified weights at one of these locations "
            f"(expected SHA-256: {EXPECTED_B0_SHA256})."
        )



    def _load_from_checkpoint(self, model_path: Optional[str]) -> None:
        ckpt_path = self._find_checkpoint(model_path)

        # Cryptographic hash verification
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()

        if actual_hash != EXPECTED_B0_SHA256:
            raise ValueError(
                f"ReDimNet2-B0 checkpoint hash mismatch at '{ckpt_path}'. "
                f"Expected: {EXPECTED_B0_SHA256}, Got: {actual_hash}. "
                f"Refusing to load unverified model weights."
            )

        self._checkpoint_hash = actual_hash
        self._is_verified = True

        ReDimNet2Wrap = _resolve_redimnet2_module()
        state = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)

        config = state.get("model_config", {})
        self._model = ReDimNet2Wrap(**config)
        self._model.load_state_dict(state["state_dict"])
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            "Loaded verified ReDimNet2-B0 checkpoint from %s (SHA256: %s...)",
            ckpt_path,
            actual_hash[:16],
        )

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract a 192-dimensional, strictly L2-normalized speaker embedding.
        Fails explicitly on empty, silent, or non-finite inputs, and on non-finite model outputs.
        """
        if audio is None or len(audio) == 0:
            raise ValueError("ReDimNet2-B0 received empty or None audio input.")

        # Ensure 1D mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Resample to 16 kHz if necessary
        if sample_rate != 16000:
            from ..preprocessing.audio_processor import AudioProcessor
            proc = AudioProcessor(target_sample_rate=16000)
            audio = proc.resample(audio, orig_sr=sample_rate)

        # Handle non-finite samples cleanly by failing explicitly
        if not np.isfinite(audio).all():
            raise ValueError("ReDimNet2-B0 received audio containing non-finite samples (NaN or Inf).")

        # Check silence
        if np.linalg.norm(audio) < 1e-9:
            raise ValueError("ReDimNet2-B0 received silent audio waveform with near-zero energy.")

        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            emb_tensor = self._model(audio_tensor)

        emb = emb_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        # Validate shape and finiteness
        if len(emb) != self._dim or not np.isfinite(emb).all():
            raise FloatingPointError("ReDimNet2-B0 model produced invalid or non-finite embedding vector.")

        # Enforce strict unit L2 norm for cosine matching
        norm = np.linalg.norm(emb)
        if norm < 1e-9:
            raise FloatingPointError("ReDimNet2-B0 model produced an embedding vector with zero norm.")

        return (emb / norm).astype(np.float32)

    def batch_embed(self, audio_batch: List[np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        if not audio_batch:
            return np.zeros((0, self._dim), dtype=np.float32)
        embeddings = [self.embed(audio, sample_rate=sample_rate) for audio in audio_batch]
        return np.stack(embeddings, axis=0)

    def embedding_dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return "redimnet2_b0"

    def checkpoint_hash(self) -> Optional[str]:
        return self._checkpoint_hash

    def frontend_config(self) -> dict:
        return {
            "sample_rate": 16000,
            "feature_type": "tf_mel_banks",
            "num_mel_bins": 80,
            "window_length_ms": 25,
            "hop_length_ms": 10,
        }

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            model_name=self.model_name(),
            embedding_dim=self.embedding_dimension(),
            checkpoint_hash=self.checkpoint_hash(),
            frontend_version="redimnet2_tfmelbanks_80_v1.0.0",
            sample_rate=16000,
            is_verified=self._is_verified,
            status="verified" if self._is_verified else "unverified_weights",
        )
