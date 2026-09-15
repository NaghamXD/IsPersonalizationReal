"""D32 run scoping must actually scope. Written after it silently did not.

A duplicate `RESULTS_DIR = OUTPUTS_DIR / "results"` later in config.py overrode the
scoped definition, so an all-data run wrote its results into the Phase 1 directory. It
was caught only because the scores came back identical to Phase 1 -- the second half of
the same bug was that scripts/score_lopo_matrix.py hardcoded the Phase 1 checkpoint path
and so had loaded the Phase 1 backbone. Either alone would have produced a plausible but
entirely wrong "all-data result".
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCOPED = ["LABELS_JSON", "FOLDS_DIR", "POOLS_DIR", "BASELINE_CKPT_ROOT",
          "HYPER_CKPT_ROOT", "SIGNATURES_DIR", "RESULTS_DIR", "THRESHOLDS_DIR"]


def _config_under(run: str) -> dict:
    """Import config in a subprocess with VSVIG_RUN set, and report the paths."""
    code = ("import json, config; "
            "print(json.dumps({k: str(getattr(config, k)) for k in %r}))" % SCOPED)
    env = {"PATH": "/usr/bin:/bin", "VSVIG_RUN": run,
           "VSVIG_DATA_ROOT": str(ROOT / "WU-SAHZU-EMU-Video" / "dataset")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    import json
    return json.loads(out.stdout)


def test_every_scoped_path_is_defined_exactly_once():
    """A later duplicate silently wins and un-scopes the path."""
    src = (ROOT / "config.py").read_text()
    for name in SCOPED:
        n = len(re.findall(rf"^{name} = ", src, re.M))
        assert n == 1, f"{name} is defined {n} times in config.py; a duplicate un-scopes it"


def test_run_name_changes_every_scoped_path():
    base, run = _config_under(""), _config_under("alldata")
    for name in SCOPED:
        assert base[name] != run[name], f"{name} did not change under VSVIG_RUN"
        assert "alldata" in run[name], f"{name} lacks the run suffix: {run[name]}"


def test_phase1_paths_are_unchanged_without_a_run_name():
    base = _config_under("")
    assert base["LABELS_JSON"] == "processed_data/labels.json"
    assert base["FOLDS_DIR"] == "processed_data/folds"
    assert base["BASELINE_CKPT_ROOT"] == "outputs/lopo/checkpoints"
    assert base["RESULTS_DIR"] == "outputs/results"


def test_no_script_hardcodes_a_run_scoped_directory():
    """Scripts must go through config, or they will read the wrong experiment."""
    bad = []
    pat = re.compile(r'["\'](?:outputs/(?:lopo|signatures|results|thresholds)'
                     r'|processed_data/(?:folds|pools)|processed_data/labels\.json)')
    for f in sorted((ROOT / "scripts").glob("*.py")):
        if f.name == "freeze_artifacts.py":      # deliberately inventories Phase 1
            continue
        for i, line in enumerate(f.read_text().split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            if pat.search(line):
                bad.append(f"{f.name}:{i}: {line.strip()[:90]}")
    assert not bad, "hardcoded run-scoped paths:\n  " + "\n  ".join(bad)
