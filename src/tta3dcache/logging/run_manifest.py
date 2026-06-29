from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


@dataclass(slots=True)
class RunManifest:
    run_id: str
    config: dict[str, Any]
    command: list[str]
    environment: dict[str, Any]


def build_manifest(run_id: str, config: dict[str, Any], command: list[str], repo_root: str | Path) -> RunManifest:
    repo_root = Path(repo_root)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "git_commit_root": _git_commit(repo_root),
        "git_commit_cdviews": _git_commit(repo_root / "src/tta3dcache/cdViews"),
        "git_commit_tda": _git_commit(repo_root / "src/tta3dcache/TDA"),
        "git_commit_uni_adapter": _git_commit(repo_root / "src/tta3dcache/Uni-Adapter"),
    }
    return RunManifest(run_id=run_id, config=config, command=command, environment=environment)


def dump_manifest(path: str | Path, manifest: RunManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, ensure_ascii=True)
