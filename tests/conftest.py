"""conftest.py – Add src/ to sys.path so tests can import agent modules."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
