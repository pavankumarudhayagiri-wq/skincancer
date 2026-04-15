"""
Fetch trained model files from the upstream GitHub repository (Git LFS).

Plain "Download ZIP" from GitHub does NOT include LFS blobs — you get tiny pointer
files instead. This script clones the repo with Git LFS and copies the weights
into this folder.

Requirements:
  - Git for Windows installed (https://git-scm.com/download/win)
  - Git LFS: after installing Git, run once:  git lfs install

Usage (from this directory):
  python download_weights.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/pavankumarudhayagiri-wq/skincancer.git"
TARGET_DIR = Path(__file__).resolve().parent

# Copy every weight artifact we expect (same names as melanoma_ml.py)
DERM = [
    "DM_melanoma_cnn_with_saliency.keras",
    "DM_vgg16_model_with_saliency.keras",
    "DM_best_ResNet50_model.keras",
    "DM_efficientnetb4_model_with_saliency.keras",
    "DM_InceptionResNetV2_model.keras",
]
SKIN = [
    "CNN_skin_classifier_weights.weights.h5",
    "best_VGG16_weights.weights.h5",
    "best_ResNet50_weights.weights.h5",
    "best_EfficientNetB4_weights.weights.h5",
    "best_InceptionResNetV2_weights.weights.h5",
]
FILES = DERM + SKIN


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {r.returncode}")


def main() -> int:
    try:
        _run(["git", "lfs", "version"])
    except FileNotFoundError:
        print("Git not found. Install Git for Windows, then re-run.", file=sys.stderr)
        return 1
    except RuntimeError:
        print("Git LFS not available. Run: git lfs install", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="melanoma_weights_") as tmp:
        clone = Path(tmp) / "repo"
        print(f"Cloning (with LFS) into temporary folder…")
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                REPO_URL,
                str(clone),
            ]
        )
        # Ensure LFS pull; some environments need explicit include paths
        include_arg = ",".join(FILES)
        pulled = False
        for cmd in (
            ["git", "lfs", "pull", "--include", include_arg],
            ["git", "lfs", "fetch", "--include", include_arg, "origin", "main"],
            ["git", "lfs", "checkout"],
            ["git", "lfs", "pull"],
        ):
            try:
                _run(cmd, cwd=clone)
                pulled = True
            except RuntimeError:
                continue
        if not pulled:
            print("Warning: Git LFS pull/fetch commands did not report success.", file=sys.stderr)

        copied = 0
        missing_in_repo: list[str] = []
        for name in FILES:
            src = clone / name
            if not src.is_file():
                missing_in_repo.append(name)
                continue
            dst = TARGET_DIR / name
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"OK {name} ({size_mb:.1f} MB)")
            copied += 1

        if missing_in_repo:
            print("Not found in repo clone:", ", ".join(missing_in_repo), file=sys.stderr)

        if copied == 0:
            print(
                "No weight files were copied. If files are ~130 bytes on disk, "
                "Git LFS did not run — install Git LFS and run: git lfs install",
                file=sys.stderr,
            )
            return 1

        print(f"\nDone. Copied {copied} file(s) to:\n  {TARGET_DIR}")
        print("Restart Streamlit and use Settings → Clear cache if models were cached empty.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
