"""
EchoLocate — Visual Demonstration & Analytics Platform Launcher
================================================================
Launches the full interactive research visualization platform for examiner defense.

Usage:
    python scripts/run_echolocate_visual_demo.py [--port 8000] [--host 127.0.0.1] [--no-browser]
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

import uvicorn

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.app.hardware.device_detector import detect_audio_input_devices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("EchoLocate.VisualDemo")


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoLocate Visual Analytics Platform Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")

    args = parser.parse_args()

    logger.info("================================================================================")
    logger.info("EchoLocate — Interactive Research Visualization Platform (Experiment 6.25)")
    logger.info("================================================================================")

    devices = detect_audio_input_devices()
    logger.info("Detected %d audio input devices.", len(devices))
    for d in devices:
        logger.info("  [%d] %s (Channels: %d, Synced: %s)", d.device_id, d.name, d.max_input_channels, d.is_synchronized_array)

    url = f"http://{args.host}:{args.port}"
    logger.info("Launching Visual Dashboard at: %s", url)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning("Could not auto-open browser: %s", e)

    uvicorn.run("src.app.main:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
