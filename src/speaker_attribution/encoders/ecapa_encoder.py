"""
EchoLocate — ECAPA-TDNN Speaker Encoder
=======================================
Adapted from 3D-Speaker & Interspeech 2020 (Desplanques et al.).
Emphasized Channel Attention, Propagation and Aggregation in TDNN-based speaker verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..preprocessing.features import compute_fbank
from ..preprocessing.vad import extract_speech_only_samples, create_sliding_windows
from .base import SpeakerEncoder

logger = logging.getLogger(__name__)


class SEModule(nn.Module):
    """Squeeze-and-Excitation Channel Attention module."""

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class SERes2NetBlock(nn.Module):
    """Res2Net convolutional block with Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, scale: int = 4):
        super().__init__()
        self.scale = scale
        width = channels // scale
        self.width = width

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                width,
                width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2,
                bias=False,
            )
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])

        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)

        self.se = SEModule(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))

        spx = torch.split(out, self.width, 1)
        y = spx[0]
        ys = [y]
        for i in range(self.scale - 1):
            if i == 0:
                sp = spx[i + 1]
            else:
                sp = spx[i + 1] + ys[-1]
            y = self.relu(self.bns[i](self.convs[i](sp)))
            ys.append(y)

        out = torch.cat(ys, 1)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.se(out)
        out += residual
        return out


class ECAPATDNNNet(nn.Module):
    """Core ECAPA-TDNN neural architecture."""

    def __init__(self, feat_dim: int = 80, channels: int = 512, embedding_dim: int = 192):
        super().__init__()
        self.feat_dim = feat_dim
        self.channels = channels
        self.embedding_dim = embedding_dim

        # Initial TDNN layer
        self.conv1 = nn.Conv1d(feat_dim, channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        # 3 Multiscale SE-Res2Net blocks with progressive dilations
        self.layer1 = SERes2NetBlock(channels, kernel_size=3, dilation=2, scale=4)
        self.layer2 = SERes2NetBlock(channels, kernel_size=3, dilation=3, scale=4)
        self.layer3 = SERes2NetBlock(channels, kernel_size=3, dilation=4, scale=4)

        # Multi-layer Feature Aggregation (MFA)
        self.mfa = nn.Conv1d(channels * 3, 1536, kernel_size=1, bias=False)
        self.bn_mfa = nn.BatchNorm1d(1536)

        # Attentive Statistics Pooling
        self.asp_att = nn.Sequential(
            nn.Conv1d(1536, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )

        # Final bottleneck projection
        self.fc = nn.Linear(1536 * 2, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[1] != self.feat_dim and x.shape[2] == self.feat_dim:
            x = x.transpose(1, 2)

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # Concat across temporal propagation layers
        mfa_in = torch.cat([x1, x2, x3], dim=1)
        fused = self.relu(self.bn_mfa(self.mfa(mfa_in)))

        # ASP
        alpha = self.asp_att(fused)
        mean = torch.sum(alpha * fused, dim=2)
        var = torch.sum(alpha * (fused ** 2), dim=2) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-5))
        stats = torch.cat([mean, std], dim=1)

        embedding = self.bn_out(self.fc(stats))
        return F.normalize(embedding, p=2, dim=1)


class ECAPAEncoder(SpeakerEncoder):
    """
    ECAPA-TDNN Speaker Encoder adapter implementing SpeakerEncoder interface.
    Integrates pre-trained SpeechBrain VoxCeleb ECAPA-TDNN model for production-grade
    speaker verification, with seamless fallback to custom PyTorch architecture.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 192,
        channels: int = 512,
        device: str = "cpu",
    ):
        self._device_str = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self._device = torch.device(self._device_str)
        self._dim = embedding_dim
        self._classifier = None

        # Attempt to load pre-trained SpeechBrain model
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            save_dir = Path("demo_data/models/speechbrain_ecapa")
            save_dir.mkdir(parents=True, exist_ok=True)
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(save_dir),
                run_opts={"device": self._device_str},
                local_strategy=LocalStrategy.COPY,
            )
            logger.info("Loaded pre-trained SpeechBrain ECAPA-TDNN VoxCeleb model")
        except Exception as e:
            logger.warning("Could not initialize SpeechBrain ECAPA: %s. Using internal architecture.", e)
            self._classifier = None

        self._model = ECAPATDNNNet(feat_dim=80, channels=channels, embedding_dim=self._dim)
        self._model.to(self._device)
        self._model.eval()

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            return
        p = Path(model_path)
        if p.exists():
            state = torch.load(str(p), map_location=self._device)
            if "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state, strict=False)
            logger.info("Loaded ECAPA-TDNN weights from %s", p)
        else:
            logger.warning("ECAPA weight path not found: %s. Using initialized model.", model_path)

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if audio is None or len(audio) == 0:
            return np.zeros(self._dim, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        # 1. Speech-only audio filtering (Phase 5)
        speech = extract_speech_only_samples(audio, sample_rate=sample_rate)
        if len(speech) == 0:
            speech = audio

        # 2. Multi-window slicing (Phase 6)
        windows = create_sliding_windows(speech, sample_rate=sample_rate, window_duration_sec=1.5, hop_duration_sec=0.75)

        window_embeddings = []
        for win in windows:
            if self._classifier is not None:
                try:
                    sig_t = torch.from_numpy(win.astype(np.float32)).unsqueeze(0).to(self._device)
                    emb = self._classifier.encode_batch(sig_t)
                    emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
                    window_embeddings.append(emb)
                    continue
                except Exception as e:
                    logger.warning("SpeechBrain ECAPA inference error: %s, falling back to internal net", e)

            feats = compute_fbank(win, sample_rate=sample_rate)
            if len(feats) == 0:
                continue

            t_feats = torch.from_numpy(feats).float().unsqueeze(0).to(self._device)
            with torch.no_grad():
                emb = self._model(t_feats)
            w_emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
            w_norm = np.linalg.norm(w_emb)
            if w_norm > 1e-8:
                w_emb = w_emb / w_norm
            window_embeddings.append(w_emb)

        if not window_embeddings:
            return np.zeros(self._dim, dtype=np.float32)

        avg_emb = np.mean(window_embeddings, axis=0)
        avg_norm = np.linalg.norm(avg_emb)
        if avg_norm > 1e-8:
            avg_emb = avg_emb / avg_norm
        return avg_emb.astype(np.float32)

    def batch_embed(self, audio_batch: List[np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        if not audio_batch:
            return np.zeros((0, self._dim), dtype=np.float32)

        embeddings = [self.embed(audio, sample_rate=sample_rate) for audio in audio_batch]
        return np.stack(embeddings, axis=0)

    def embedding_dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return "ecapa"
