"""Legacy Streamlit entrypoint that directly loads `melanoma_ml.py`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Melanoma Detection App", page_icon="🔬", layout="wide")

_APP_DIR = Path(__file__).resolve().parent
_ml_err: Exception | None = None
_melanoma_ml = None

try:
    _spec = importlib.util.spec_from_file_location("melanoma_ml", _APP_DIR / "melanoma_ml.py")
    if _spec is None or _spec.loader is None:
        raise ImportError("Could not load melanoma_ml.py")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _melanoma_ml = _module
except Exception as exc:
    _ml_err = exc


def run_app() -> None:
    if _ml_err is not None or _melanoma_ml is None:
        st.error("The app could not start because melanoma_ml.py failed to load.")
        st.exception(_ml_err)
        return
    _melanoma_ml.run_app()


if __name__ == "__main__":
    run_app()
