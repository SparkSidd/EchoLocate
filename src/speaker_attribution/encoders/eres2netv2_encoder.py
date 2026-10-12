"""
EchoLocate — ERes2NetV2 Speaker Encoder
=======================================
Adapted from 3D-Speaker & Interspeech 2024.
ERes2NetV2: Efficient multi-scale residual network with Local & Global Feature Fusion (LGFF)
optimized for short-duration speaker recognition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..preprocessing.features import compute_fbank, compute_kaldi_fbank
from ..preprocessing.vad import extract_speech_only_samples, create_sliding_windows
from .base import SpeakerEncoder

logger = logging.getLogger(__name__)


class Res2NetBlock(nn.Module):
    """Pruned multi-scale Res2Net convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        width = out_channels // scale
        self.width = width

        self.conv1 = nn.Conv1d(in_channels, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(width) for _ in range(scale - 1)
        ])

        self.conv3 = nn.Conv1d(width * scale, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
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
        out = self.bn3(self.conv3(out))
        out += residual
        return self.relu(out)


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling (mean and weighted standard deviation)."""

    def __init__(self, in_dim: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, attention_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attention_dim),
            nn.Conv1d(attention_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        alpha = self.attention(x)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(torch.clamp(residuals, min=1e-5))
        return torch.cat([mean, std], dim=1)


class ERes2NetV2Net(nn.Module):
    """Core ERes2NetV2 neural architecture."""

    def __init__(self, feat_dim: int = 80, embedding_dim: int = 192):
        super().__init__()
        self.feat_dim = feat_dim
        self.embedding_dim = embedding_dim

        self.in_conv = nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

        # Multi-scale blocks
        self.stage1 = Res2NetBlock(128, 128, scale=4)
        self.stage2 = Res2NetBlock(128, 256, scale=4)
        self.stage3 = Res2NetBlock(256, 384, scale=4)

        # Local & Global Feature Fusion (LGFF)
        self.fusion = nn.Conv1d(128 + 256 + 384, 512, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm1d(512)

        # Attentive Stats Pooling
        self.asp = AttentiveStatsPooling(512, attention_dim=128)

        # Output linear bottleneck
        self.fc = nn.Linear(512 * 2, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[1] != self.feat_dim and x.shape[2] == self.feat_dim:
            x = x.transpose(1, 2)

        x0 = self.relu(self.in_bn(self.in_conv(x)))
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        # LGFF concat
        fused = torch.cat([x1, x2, x3], dim=1)
        fused = self.relu(self.bn_fusion(self.fusion(fused)))

        pooled = self.asp(fused)
        embedding = self.bn_out(self.fc(pooled))
        return F.normalize(embedding, p=2, dim=1)


class ERes2NetV2Encoder(SpeakerEncoder):
    """
    ERes2NetV2 Speaker Encoder adapter implementing SpeakerEncoder interface.
    Integrates pre-trained 3D-Speaker ERes2Net model for production-grade
    speaker verification, with seamless fallback to custom PyTorch architecture.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 512,
        device: str = "cpu",
    ):
        self._device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self._dim = embedding_dim
        self._onnx_sess = None

        # Attempt to load pre-trained ONNX model if available and dimension matches (512)
        onnx_path = Path("demo_data/models/eres2net/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
        if onnx_path.exists() and self._dim == 512:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self._onnx_sess = ort.InferenceSession(str(onnx_path), sess_options=opts)
                logger.info("Loaded pre-trained ERes2Net ONNX model from %s", onnx_path)
            except Exception as e:
                logger.warning("Could not initialize ERes2Net ONNX: %s. Using internal architecture.", e)
                self._onnx_sess = None

        self._model = ERes2NetV2Net(feat_dim=80, embedding_dim=self._dim)
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
            logger.info("Loaded ERes2NetV2 weights from %s", p)
        else:
            logger.warning("ERes2NetV2 weight path not found: %s. Using initialized model.", model_path)

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if audio is None or len(audio) == 0:
            return np.zeros(self._dim, dtype=np.float32)

        # 1. Strip long silence to extract speech-only audio (Phase 5)
        speech = extract_speech_only_samples(audio, sample_rate=sample_rate)
        if len(speech) == 0:
            speech = audio

        # 2. Slice into multi-window chunks if sufficient duration (Phase 6)
        windows = create_sliding_windows(speech, sample_rate=sample_rate, window_duration_sec=1.5, hop_duration_sec=0.75)
        
        window_embeddings = []
        for win in windows:
            # Official 3D-Speaker Kaldi FBank (Phase 1)
            feats = compute_kaldi_fbank(win, sample_rate=sample_rate)
            if len(feats) == 0:
                continue

            if self._onnx_sess is not None:
                try:
                    feats_in = np.expand_dims(feats, axis=0).astype(np.float32)
                    emb = self._onnx_sess.run(None, {"x": feats_in})[0].squeeze()
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
                    window_embeddings.append(emb.astype(np.float32))
                    continue
                except Exception as e:
                    logger.warning("ERes2Net ONNX inference error: %s, falling back to internal net", e)

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

        # Multi-window L2-normalized mean aggregation (Phase 6)
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
        return "eres2netv2"
