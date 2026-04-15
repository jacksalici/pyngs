import sys
import os

# Add the cloned pyngs source to the Python path
_pyngs_src = os.path.join(os.path.dirname(__file__), "src")
if _pyngs_src not in sys.path:
    sys.path.insert(0, _pyngs_src)

from pyngs import Pyngs, Config, Logger, ShapeHook

__all__ = ["Pyngs", "Config", "Logger", "ShapeHook"]
