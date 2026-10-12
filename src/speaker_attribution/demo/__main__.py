"""
EchoLocate — Demo execution entry point.
Run via:
    python -m src.speaker_attribution.demo enroll
    python -m src.speaker_attribution.demo identify
    python -m src.speaker_attribution.demo blind-test
    python -m src.speaker_attribution.demo web
"""

from .cli import main

if __name__ == "__main__":
    main()
