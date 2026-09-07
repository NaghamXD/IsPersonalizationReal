"""Write the exact environment behind a set of results.

environment.yml and requirements.txt say how to BUILD an environment. This says what
was actually RUNNING, which is the thing a result has to be reproducible from.

Conda-aware: `pip freeze` alone on a conda env records only the pip-installed subset
and nothing about channels, so the lock would look complete while being wrong -- worse
than having none. When CONDA_PREFIX is set we export both.

    python scripts/capture_env.py
"""
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def header():
    lines = [
        f"# captured: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        f"# python: {sys.version.split()[0]}  ({sys.executable})",
        f"# platform: {platform.platform()}  {platform.machine()}",
    ]
    conda = os.environ.get("CONDA_PREFIX")
    lines.append(f"# conda_prefix: {conda or 'NOT A CONDA ENV'}")
    if conda:
        lines.append(f"# conda_env: {os.environ.get('CONDA_DEFAULT_ENV', '?')}")
    try:
        import torch
        lines += [
            f"# torch: {torch.__version__}",
            f"# mps_available: {torch.backends.mps.is_available()}",
            f"# mps_built: {torch.backends.mps.is_built()}",
            f"# cuda_available: {torch.cuda.is_available()}",
        ]
        if not torch.backends.mps.is_available():
            lines.append("# WARNING: MPS unavailable -- training will fall back to CPU")
    except ImportError:
        lines.append("# torch: NOT INSTALLED")
    git = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    lines.append(f"# git_commit: {(git or 'unknown').strip()}")
    return "\n".join(lines) + "\n"


def main():
    wrote = []
    h = header()

    freeze = run([sys.executable, "-m", "pip", "freeze"])
    if freeze is not None:
        p = ROOT / "environment_lock.txt"
        p.write_text(h + freeze)
        wrote.append(p)

    if os.environ.get("CONDA_PREFIX"):
        # --no-builds keeps the export portable across machines; the build strings are
        # platform-specific and would make the file unusable anywhere else.
        exported = run("conda env export --no-builds")
        if exported:
            p = ROOT / "environment_lock.yml"
            p.write_text(h + exported)
            wrote.append(p)
        else:
            print("  [warn] `conda env export` failed -- is conda on PATH in this shell?")
    else:
        print("  [note] CONDA_PREFIX not set; captured pip only. If you meant to be in "
              "a conda env, activate it and re-run.")

    print(h.rstrip())
    for p in wrote:
        print(f"wrote {p}")
    if not wrote:
        print("nothing captured -- neither pip freeze nor conda export succeeded")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
