"""
EchoLocate — Real Hardware & Smartphone Client API Router
=========================================================
REST and WebSocket endpoints supporting smartphone audio streaming, NTP clock probing,
acoustic calibration, live TDOA/hyperbolic locus querying, and anti-fake reprocessing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.app.engine.hardware_pipeline import HardwareMode, LiveHardwarePipeline

logger = logging.getLogger(__name__)


class DevicePositionUpdate(BaseModel):
    device_id: str
    x: float
    y: float


class ModeSwitchRequest(BaseModel):
    mode: str


class CalibrationRequest(BaseModel):
    source_x: float = 3.0
    source_y: float = 3.0


class NTPProbeRequest(BaseModel):
    device_id: str
    t0_client_send_ms: float


def create_hardware_router(pipeline: LiveHardwarePipeline) -> APIRouter:
    """Create FastAPI router for hardware operations."""
    router = APIRouter(prefix="/api/hardware", tags=["Hardware Demonstration"])

    @router.get("/registry_devices")
    async def list_hardware_devices() -> list[dict[str, Any]]:
        """List all physical and network devices in multi-device registry."""
        return [d.to_dict() for d in pipeline.registry.list_devices()]

    @router.post("/position")
    async def update_device_coordinate(req: DevicePositionUpdate) -> dict[str, Any]:
        """Update geometric position of a device."""
        ok = pipeline.registry.update_position(req.device_id, req.x, req.y)
        return {"success": ok, "device_id": req.device_id, "position": [req.x, req.y]}

    @router.post("/mode")
    async def set_hardware_mode(req: ModeSwitchRequest) -> dict[str, Any]:
        """Switch between Single-Laptop (H1) and Dual-Device (H2)."""
        try:
            mode_enum = HardwareMode(req.mode)
            pipeline.set_mode(mode_enum)
            return {"success": True, "mode": mode_enum.value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @router.post("/calibrate")
    async def run_acoustic_calibration(req: CalibrationRequest) -> dict[str, Any]:
        """Trigger physical acoustic calibration."""
        res = pipeline.trigger_acoustic_calibration(source_xy=(req.source_x, req.source_y))
        return res.to_dict()

    @router.post("/calibrate_gain")
    async def calibrate_microphone_sensitivity() -> dict[str, Any]:
        """Auto-calibrate hardware gain difference when sound source is equidistant in the center."""
        offset_db = pipeline.calibrate_gain()
        return {"success": True, "mic_gain_offset_db": round(offset_db, 2)}

    @router.get("/status")
    async def get_hardware_status() -> dict[str, Any]:
        """Fetch current hardware pipeline frame output."""
        frame = pipeline.process_cycle()
        return frame.to_dict()

    @router.post("/probe")
    async def ntp_clock_probe(req: NTPProbeRequest) -> dict[str, Any]:
        """NTP-style clock probe endpoint for remote clients."""
        t1_recv = time.time() * 1000.0
        t2_send = time.time() * 1000.0
        # Client sends t3, then completes probe
        return {
            "device_id": req.device_id,
            "t0_client_send_ms": req.t0_client_send_ms,
            "t1_server_recv_ms": t1_recv,
            "t2_server_send_ms": t2_send,
        }

    @router.post("/reprocess")
    async def reprocess_frame() -> dict[str, Any]:
        """⚡ CAPTURE AND REPROCESS verification action."""
        frame = pipeline.reprocess_current_frame()
        return {
            "reprocessed": True,
            "frame_id": frame.frame_id,
            "computation_hash": frame.computation_hash,
            "tdoa_ms": frame.tdoa_estimate.tdoa_ms if frame.tdoa_estimate else None,
            "latency_ms": frame.processing_latency_ms,
        }

    return router
