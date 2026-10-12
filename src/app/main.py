"""
EchoLocate — Production Demonstration Application Server
========================================================
Main FastAPI application entry point serving the interactive web GUI,
REST API routes, and real-time WebSocket localization streams.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.app.api.routes import create_api_router
from src.app.api.websocket_manager import WebSocketConnectionManager
from src.app.engine.method_registry import EchoLocateMethodRegistry
from src.app.engine.pipeline_worker import RealTimePipelineWorker
from src.app.engine.scenario_manager import PresetScenarioManager
from src.app.engine.temporal_tracker import CausalTemporalTracker
from src.app.state.app_state import ApplicationState

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the EchoLocate FastAPI application."""
    app = FastAPI(
        title="EchoLocate Demonstration Application",
        description="Real-Time Acoustic Source Localization System (Experiment 6)",
        version="6.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Core Engine & State
    app_state = ApplicationState()
    registry = EchoLocateMethodRegistry(
        room_dimensions_m=app_state.room_dimensions_m,
        mic_positions_2d=None,
    )
    tracker = CausalTemporalTracker(mode="CAUSAL_SMOOTHED")
    worker = RealTimePipelineWorker(
        method_registry=registry,
        temporal_tracker=tracker,
        active_method=app_state.active_method,
    )
    scenario_mgr = PresetScenarioManager()
    ws_manager = WebSocketConnectionManager()

    # Bridge worker results to application state and WebSocket
    worker.register_result_callback(lambda packet: app_state.update_result(packet))
    worker.start()

    # Mount UI Static Assets
    ui_dir = Path(__file__).resolve().parent / "ui"
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

    # Register API Endpoints
    router = create_api_router(app_state, registry, scenario_mgr)
    app.include_router(router)

    from src.app.api.visualization_routes import create_visualization_router
    vis_router = create_visualization_router(app_state, registry, scenario_mgr)
    app.include_router(vis_router)

    from src.app.engine.hardware_pipeline import LiveHardwarePipeline
    from src.app.api.hardware_routes import create_hardware_router
    hardware_pipeline = LiveHardwarePipeline(sample_rate=registry.fs, room_dims=registry.room_dims)
    hw_router = create_hardware_router(hardware_pipeline)
    app.include_router(hw_router)

    @app.get("/")
    async def serve_index() -> FileResponse:
        """Serve main interactive dashboard."""
        index_file = ui_dir / "index.html"
        return FileResponse(str(index_file))

    @app.get("/phone")
    async def serve_phone_client() -> FileResponse:
        """Serve smartphone Web Audio client page."""
        phone_file = ui_dir / "phone.html"
        return FileResponse(str(phone_file))

    @app.websocket("/ws/stream")
    async def websocket_stream_endpoint(websocket: WebSocket) -> None:
        """WebSocket stream for real-time localization updates."""
        await ws_manager.connect(websocket)
        try:
            while True:
                msg = await websocket.receive_text()
        except Exception:
            ws_manager.disconnect(websocket)

    @app.websocket("/ws/phone_audio")
    async def websocket_phone_audio_endpoint(websocket: WebSocket) -> None:
        """WebSocket stream receiving live audio from smartphone client."""
        await websocket.accept()
        dev_id = "smartphone_client_primary"
        session = hardware_pipeline.network_receiver.get_or_create_session(dev_id, client_ip=websocket.client.host if websocket.client else "127.0.0.1")
        session.is_streaming = True
        dev = hardware_pipeline.registry.get_device(dev_id)
        if dev:
            dev.is_active = True

        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"]:
                    data_str = message["text"]
                    if data_str.startswith("{"):
                        # Check if NTP probe or audio JSON
                        try:
                            payload = json.loads(data_str)
                            if payload.get("type") == "ntp_probe":
                                t1 = time.time() * 1000.0
                                resp = {
                                    "type": "ntp_reply",
                                    "t0": payload["t0"],
                                    "t1": t1,
                                    "t2": time.time() * 1000.0,
                                }
                                await websocket.send_text(json.dumps(resp))
                            elif payload.get("type") == "ntp_finish":
                                hardware_pipeline.clock_sync.process_probe(
                                    device_id=dev_id,
                                    t0_client_send_ms=payload["t0"],
                                    t1_server_recv_ms=payload["t1"],
                                    t2_server_send_ms=payload["t2"],
                                    t3_client_recv_ms=payload["t3"],
                                )
                            else:
                                hardware_pipeline.network_receiver.decode_json_message(session, data_str)
                        except Exception:
                            pass
                elif "bytes" in message and message["bytes"]:
                    hardware_pipeline.network_receiver.decode_binary_message(session, message["bytes"])
        except (WebSocketDisconnect, Exception):
            session.is_streaming = False
            if dev:
                dev.is_active = False

    return app


app = create_app()
