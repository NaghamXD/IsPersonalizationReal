"""Run manifests. Every artifact this project writes gets one beside it.

A checkpoint with no manifest cannot be trusted later: you cannot tell which cohort
definition, which Pool A size, or which loss produced it. That is exactly the problem
we hit auditing the previous run directory.
"""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_state(repo_root: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _config_snapshot() -> dict:
    """Every public constant in config.py, so a run records the spec it ran under."""
    import config
    out = {}
    for k in dir(config):
        if k.startswith("_") or k.isupper() is False:
            continue
        v = getattr(config, k)
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def write_manifest(out_path: Path, *, seed: int, extra: dict | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "config": _config_snapshot(),
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["device"] = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    except ImportError:
        manifest["torch"] = None
    if extra:
        manifest["run"] = extra
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    return out_path
