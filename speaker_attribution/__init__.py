"""
EchoLocate — Root Shim for speaker_attribution
==============================================
Enables `import speaker_attribution` directly from workspace root.
"""

import sys
from pathlib import Path

# Ensure src/ is in sys.path
_src_dir = Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from src.speaker_attribution import *
