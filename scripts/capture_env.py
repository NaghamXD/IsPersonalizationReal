"""Write environment_lock.txt with the exact resolved versions on THIS machine."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "environment_lock.txt"
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                        capture_output=True, text=True).stdout
header = [f"# python: {sys.version.split()[0]}"]
try:
    import torch
    header.append(f"# torch: {torch.__version__}")
    header.append(f"# mps_available: {torch.backends.mps.is_available()}")
    header.append(f"# cuda_available: {torch.cuda.is_available()}")
except ImportError:
    header.append("# torch: NOT INSTALLED")
out.write_text("\n".join(header) + "\n" + freeze)
print(f"wrote {out}")
