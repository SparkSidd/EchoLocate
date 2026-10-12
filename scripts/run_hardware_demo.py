"""
EchoLocate — Real Hardware Live Demonstration Launcher (Experiment 6.5)
=======================================================================
Runs HTTP (port 8000) and HTTPS (port 8443) concurrently in a single asyncio
event loop for 100% synchronized shared in-memory hardware pipeline state.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys
import webbrowser
from pathlib import Path

import uvicorn

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.app.hardware.device_detector import detect_audio_input_devices
from src.app.main import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("EchoLocate.HardwareDemo")


def get_all_ips() -> list[str]:
    """Retrieve all local network IPv4 addresses."""
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    for known in ["192.168.137.1", "10.50.101.178"]:
        if known not in ips:
            ips.append(known)
    return ips


async def run_dual_servers(host: str = "0.0.0.0") -> None:
    """Run HTTP and HTTPS servers in the exact same event loop."""
    cert_path = root_dir / "certs" / "cert.pem"
    key_path = root_dir / "certs" / "key.pem"

    config_http = uvicorn.Config(app, host=host, port=8000, log_level="info")
    server_http = uvicorn.Server(config_http)

    tasks = [server_http.serve()]

    if cert_path.exists() and key_path.exists():
        config_https = uvicorn.Config(
            app,
            host=host,
            port=8443,
            ssl_keyfile=str(key_path),
            ssl_certfile=str(cert_path),
            log_level="info",
        )
        server_https = uvicorn.Server(config_https)
        tasks.append(server_https.serve())

    await asyncio.gather(*tasks)


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoLocate Hardware Demo Launcher")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    ips = get_all_ips()
    logger.info("================================================================================")
    logger.info("🚀 EchoLocate — Real Hardware Dual Server Running Live (Exp 6.5)")
    logger.info("================================================================================")
    logger.info("💻 Main Laptop Dashboard (HTTP):")
    logger.info("   👉 http://127.0.0.1:8000")
    logger.info("--------------------------------------------------------------------------------")
    logger.info("📱 Smartphone Microphone Client (HTTPS — Secure Context for Web Audio API):")
    for ip in ips:
        logger.info("   👉 https://%s:8443/phone", ip)
    logger.info("================================================================================")

    devices = detect_audio_input_devices()
    logger.info("Detected %d physical audio input devices.", len(devices))
    for d in devices:
        logger.info("  [%d] %s (Channels: %d)", d.device_id, d.name, d.max_input_channels)

    if not args.no_browser:
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    try:
        asyncio.run(run_dual_servers())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down EchoLocate hardware server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
