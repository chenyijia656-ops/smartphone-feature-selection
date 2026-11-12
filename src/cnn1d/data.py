"""Load the nine time-domain channels of the UCI HAR dataset."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "UCI HAR Dataset"
CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signals (N, 9, 128), zero-based labels (N,), subjects (N,)."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    base = Path(data_dir) / split
    signals = np.stack(
        [np.loadtxt(base / "Inertial Signals" / f"{name}_{split}.txt", dtype=np.float32)
         for name in CHANNELS],
        axis=1,
    )
    labels = np.loadtxt(base / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(base / f"subject_{split}.txt", dtype=np.int64)
    if signals.ndim != 3 or signals.shape[1:] != (9, 128):
        raise ValueError(f"Unexpected signal shape: {signals.shape}")
    if labels.shape != subjects.shape or len(labels) != len(signals):
        raise ValueError("Signals, labels, and subjects have different sample counts")
    if not np.all((0 <= labels) & (labels < 6)):
        raise ValueError("Labels must be in the original range 1..6")
    return signals, labels, subjects


if __name__ == "__main__":
    for split_name in ("train", "test"):
        x, y, subject = load_split(split_name)
        print(f"{split_name}: signals={x.shape}, labels={y.shape}, subjects={subject.shape}")
