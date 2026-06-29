from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PACKAGE = ROOT / "src" / "tta3dcache"
__path__ = [str(SRC_PACKAGE)] if SRC_PACKAGE.is_dir() else []

try:
    from src.tta3dcache import __version__ as __version__  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for source-only execution
    __version__ = "0.1.0"
