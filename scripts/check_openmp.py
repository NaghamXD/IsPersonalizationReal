"""Detect duplicate OpenMP runtimes before they corrupt anything.

Importing a conda-forge numeric stack alongside a pip torch loads two copies of
libomp into one process. On macOS that usually aborts outright ("OMP: Error #15"),
which is the lucky case -- the unlucky one is the documented KMP_DUPLICATE_LIB_OK
escape hatch, which lets the process continue and can "silently produce incorrect
results". Silently wrong numerics in a seizure-detection pipeline is precisely the
failure you would never catch by reading the outputs.

This imports the stack one library at a time, printing the OpenMP images loaded after
each step. If the process dies, the last line printed names the import that
introduced the second copy.

    python scripts/check_openmp.py
"""
import ctypes
import sys


def loaded_images():
    """Every dylib currently mapped into this process (macOS)."""
    try:
        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        return [libc._dyld_get_image_name(i).decode(errors="replace")
                for i in range(libc._dyld_image_count())]
    except Exception:
        return []


def omp_images():
    return sorted({p for p in loaded_images()
                   if "omp" in p.rsplit("/", 1)[-1].lower()})


def step(label, importer):
    print(f"importing {label} ...", flush=True)
    try:
        importer()
    except ImportError as e:
        print(f"  NOT INSTALLED: {e}", flush=True)
        return
    for p in omp_images():
        print(f"    omp: {p}", flush=True)


def main():
    if sys.platform != "darwin":
        print("This check is macOS-specific; on Linux the duplicate-OpenMP abort "
              "does not present the same way.")
    print(f"python: {sys.executable}\n")

    step("numpy", lambda: __import__("numpy"))
    step("scipy", lambda: __import__("scipy.linalg", fromlist=["linalg"]))
    step("cv2", lambda: __import__("cv2"))
    step("torch", lambda: __import__("torch"))

    found = omp_images()
    print("\n--- result ---")
    for p in found:
        print(f"  {p}")
    if len(found) > 1:
        print(f"\nFAIL: {len(found)} OpenMP runtimes loaded. Do NOT set "
              f"KMP_DUPLICATE_LIB_OK -- it can silently corrupt results.\n"
              f"Rebuild the env so every library comes from one source:\n"
              f"  conda env remove -n ispersonalizationreal\n"
              f"  conda env create -f environment.yml")
        return 1
    print(f"\nOK: {len(found)} OpenMP runtime loaded.")

    try:
        import torch
        print(f"torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}"
              f"  mps_built={torch.backends.mps.is_built()}")
        if not torch.backends.mps.is_available():
            print("WARNING: MPS unavailable -- training would fall back to CPU.")
    except ImportError:
        print("torch not installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
