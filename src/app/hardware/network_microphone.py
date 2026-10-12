"""
EchoLocate — Network Microphone Session & Stream Manager
=========================================================
Manages incoming smartphone Web Audio streams, packet reception, timestamping,
sequence verification, and frame queue management.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NetworkAudioPacket:
    """Individual audio packet received over WebSocket from network client."""
    device_id: str
    sequence_id: int
    client_timestamp_ms: float
    server_timestamp_ms: float
    sample_rate: int
    samples: np.ndarray  # Shape: (n_samples,) Float32
    duration_ms: float


@dataclass
class NetworkClientSession:
    """Session state for a connected smartphone or remote microphone client."""
    device_id: str
    client_ip: str
    user_agent: str
    connected_at: float = field(default_factory=time.time)
    last_packet_at: float = field(default_factory=time.time)
    packets_received: int = 0
    packets_dropped: int = 0
    expected_sequence_id: int = 0
    buffer_queue: collections.deque[NetworkAudioPacket] = field(default_factory=lambda: collections.deque(maxlen=100))
    is_streaming: bool = False

    def push_packet(self, packet: NetworkAudioPacket) -> None:
        """Push a newly received packet and audit sequence integrity."""
        self.last_packet_at = time.time()
        self.packets_received += 1

        if self.packets_received > 1:
            seq_gap = packet.sequence_id - self.expected_sequence_id
            if seq_gap > 0:
                self.packets_dropped += seq_gap
                logger.warning("Packet drop detected on %s: %d frames lost", self.device_id, seq_gap)

        self.expected_sequence_id = packet.sequence_id + 1
        self.buffer_queue.append(packet)

    def pop_latest_samples(self, target_samples: int = 3200) -> np.ndarray:
        """Extract continuous audio samples from buffer queue, padding if necessary."""
        if not self.buffer_queue:
            return np.zeros(target_samples, dtype=np.float32)

        collected = []
        needed = target_samples
        # Pop from recent packets
        for pkt in list(self.buffer_queue)[::-1]:
            s = pkt.samples
            if len(s) >= needed:
                collected.insert(0, s[-needed:])
                needed = 0
                break
            else:
                collected.insert(0, s)
                needed -= len(s)

        if collected:
            arr = np.concatenate(collected)
            if len(arr) < target_samples:
                # Zero-pad front
                pad = np.zeros(target_samples - len(arr), dtype=np.float32)
                arr = np.concatenate([pad, arr])
            return arr.astype(np.float32)

        return np.zeros(target_samples, dtype=np.float32)
