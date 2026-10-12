"""
EchoLocate — CAM++ Speaker Encoder
==================================
Adapted from 3D-Speaker (ModelScope) & Interspeech 2023.
Context-Aware Masking (CAM++) with Dense D-TDNN and Multi-Head Attention Pooling.
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


class CAMModule(nn.Module):
    """Context-Aware Masking module to adaptively weight temporal features."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        b, c, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class DTDNNBlock(nn.Module):
    """Dense Dilated TDNN block with Context-Aware Masking."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cam = CAMModule(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        out = self.cam(out)
        return torch.cat([x, out], dim=1)


class MultiQueryMultiHeadAttentionPooling(nn.Module):
    """Multi-Query Multi-Head Attention Pooling for variable-length utterance reduction."""

    def __init__(self, in_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.head_dim = in_dim // num_heads
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.key = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.value = nn.Conv1d(in_dim, in_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, t = x.shape
        k = self.key(x).view(b, self.num_heads, self.head_dim, t)
        v = self.value(x).view(b, self.num_heads, self.head_dim, t)
        q = self.query.view(1, self.num_heads, 1, self.head_dim)

        scores = torch.matmul(q, k) / (self.head_dim ** 0.5)  # (B, H, 1, T)
        weights = F.softmax(scores, dim=-1)

        pooled = torch.matmul(weights, v.transpose(-1, -2))  # (B, H, 1, head_dim)
        return pooled.view(b, c)


class CAMPPNet(nn.Module):
    """Core CAM++ neural network architecture."""

    def __init__(self, feat_dim: int = 80, embedding_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.embedding_dim = embedding_dim

        # Input projection
        self.in_conv = nn.Conv1d(feat_dim, 256, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

        # Dense D-TDNN stages
        self.block1 = DTDNNBlock(256, 128, dilation=1)
        self.block2 = DTDNNBlock(384, 128, dilation=2)
        self.block3 = DTDNNBlock(512, 128, dilation=3)

        # Transition
        self.trans = nn.Conv1d(640, 512, kernel_size=1, bias=False)
        self.trans_bn = nn.BatchNorm1d(512)

        # Multi-Head Attention Pooling
        self.pooling = MultiQueryMultiHeadAttentionPooling(512, num_heads=8)

        # Embedding projection
        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> permute to (B, F, T)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[1] != self.feat_dim and x.shape[2] == self.feat_dim:
            x = x.transpose(1, 2)

        out = self.relu(self.in_bn(self.in_conv(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.trans_bn(self.trans(out)))

        pooled = self.pooling(out)
        embedding = self.bn_out(self.fc(pooled))
        # L2 Normalization
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class CAMPPEncoder(SpeakerEncoder):
    """
    CAM++ Speaker Encoder adapter implementing SpeakerEncoder interface.
    Integrates pre-trained Alibaba DAMO / 3D-Speaker CAM++ model for production-grade
    speaker verification, with seamless fallback to custom PyTorch architecture.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 192,
        device: str = "cpu",
    ):
        self._device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self._dim = embedding_dim
        self._checkpoint_hash: Optional[str] = None
        self._onnx_sess = None

        # Attempt to load pre-trained ONNX model if available and dimension matches (192)
        onnx_path = Path("demo_data/models/campplus/campplus_cn_en_common_200k.onnx")
        if onnx_path.exists() and self._dim == 192:
            try:
                import hashlib
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self._onnx_sess = ort.InferenceSession(str(onnx_path), sess_options=opts)
                with open(onnx_path, "rb") as f:
                    self._checkpoint_hash = hashlib.sha256(f.read()).hexdigest()
                logger.info("Loaded pre-trained CAM++ ONNX model from %s", onnx_path)
            except Exception as e:
                logger.warning("Could not initialize CAM++ ONNX: %s. Using internal architecture.", e)
                self._onnx_sess = None
                self._checkpoint_hash = None


        self._model = CAMPPNet(feat_dim=80, embedding_dim=self._dim)
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
            logger.info("Loaded CAM++ weights from %s", p)
        else:
            logger.warning("CAM++ weight path not found: %s. Using initialized model.", model_path)

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
                    emb = self._onnx_sess.run(None, {"feats": feats_in})[0].squeeze()
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
                    window_embeddings.append(emb.astype(np.float32))
                    continue
                except Exception as e:
                    logger.warning("CAM++ ONNX inference error: %s, falling back to internal net", e)

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
        return "campp"

    def checkpoint_hash(self) -> Optional[str]:
        return self._checkpoint_hash
