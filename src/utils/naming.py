"""Clip-name parsing. Deliberately free of torch.

Evaluation, split construction and pool construction all need to read a clip's
patient, source event and start time, and none of them should have to import a deep
learning framework to do it.

Format, written by preprocess.py:  {patient}_{event}_{t_start_seconds}
  e.g. "pat01_Sz1_1780"  ->  ("pat01", "Sz1", 1780.0)
       "pat03_free_25"   ->  ("pat03", "free", 25.0)

The trailing field is SECONDS from the start of the source video -- `int(round(t))`,
not a frame index. Reading it as frames and dividing by an assumed frame rate is the
bug that corrupted every latency figure in the 6fca412 evaluator.
"""


def parse_clip_name(name: str) -> tuple[str, str, float]:
    parts = str(name).split("_")
    patient = parts[0]
    event = parts[1] if len(parts) > 2 else "unknown"
    try:
        t_start_s = float(parts[-1])
    except ValueError:
        t_start_s = 0.0
    return patient, event, t_start_s


def source_id(name: str) -> str:
    """'pat01_Sz1_1780' -> 'pat01_Sz1' -- the recording a clip came from."""
    patient, event, _ = parse_clip_name(name)
    return f"{patient}_{event}"


def patient_of(name: str) -> str:
    return parse_clip_name(name)[0]
