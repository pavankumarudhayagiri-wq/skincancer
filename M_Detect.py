"""Streamlit entry point: configures the page, then loads the TensorFlow app from melanoma_ml.py."""

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
        st.error("TensorFlow could not load, so the full app cannot start.")
        st.exception(_ml_err)
        st.markdown(
            """
### Why this happens

TensorFlow uses native DLLs. On Windows this often fails if **Visual C++ Redistributable** is missing,
or if the **CPU does not support AVX** instructions (TensorFlow 2.x expects them).

### What to do

1. Install **Microsoft Visual C++ Redistributable (x64)** and restart:
   https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Open the app in **Chrome or Edge** at `http://localhost:8501`.  
   The built-in Cursor/VS Code preview sometimes shows a **blank page** because Streamlit needs WebSockets.

3. If it still fails after VC++, the machine may need an older TensorFlow build or a CPU with AVX support.
            """
        )
        return

    _melanoma_ml.run_app()


if __name__ == "__main__":
    run_app()
