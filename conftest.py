# conftest.py
# Adds src/ to sys.path for all tests in this project.
# This replaces the per-file sys.path.insert(...) calls.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

