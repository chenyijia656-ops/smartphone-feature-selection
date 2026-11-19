"""Five-fold subject-disjoint validation for the time-series Transformer."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

# Reuse the CNN experiment's data, split checks, loaders, deterministic setup,
# validation calculation, and curve writer without copying their implementations.
CNN_DIR = Path(__file__).resolve().parents[1] / "cnn1d"
sys.path.insert(0, str(CNN_DIR))
from data import ROOT, load_split  # noqa: E402
from train import evaluate, load_folds, make_loader, save_curve, set_seed  # noqa: E402

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn

from model import LightweightTimeSeriesTransformer


OUTPUT_DIR = ROOT / "results" / "transformer"
SPLITS_FILE = ROOT / "results" / "grrf_cv5_splits.csv"
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2


def train_fold(fold: int, signals: np.ndarray, labels: np.ndarray,
               subjects: np.ndarray, folds: np.ndarray,
               args: argparse.Namespace, device: torch.device) -> dict:
    train_idx = np.flatnonzero(folds != fold)
    val_idx = np.flatnonzero(folds == fold)
    assert set(subjects[train_idx]).isdisjoint(set(subjects[val_idx]))
    fold_seed = args.seed + fold
    set_seed(fold_seed)
    train_loader = make_loader(signals[train_idx], labels[train_idx],
                               args.batch_size, shuffle=True, seed=fold_seed)
    val_loader = make_loader(signals[val_idx], labels[val_idx],
                             args.batch_size, seed=fold_seed)
    model = LightweightTimeSeriesTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_f1, stale, best_state = -1.0, 0, None
    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)

        val_pred, val_loss = evaluate(model, val_loader, device, criterion)
        val_f1 = f1_score(labels[val_idx], val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_idx),
                        "val_loss": val_loss,
                        "val_accuracy": accuracy_score(labels[val_idx], val_pred),
                        "val_macro_f1": val_f1})
        print(f"fold {fold} epoch {epoch:02d}: train_loss={history[-1]['train_loss']:.4f}, "
              f"val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    val_pred, _ = evaluate(model, val_loader, device, criterion)
    training_seconds = time.perf_counter() - start
    best_epoch = int(np.argmax([row["val_macro_f1"] for row in history]) + 1)
    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, fold_dir / "model.pt")
    with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_curve(history, best_epoch, fold_dir / "training_curves.png")
    np.savetxt(fold_dir / "confusion_matrix.csv",
               confusion_matrix(labels[val_idx], val_pred, labels=np.arange(6)),
               fmt="%d", delimiter=",")

    metrics = {"fold": fold, "accuracy": accuracy_score(labels[val_idx], val_pred),
               "macro_f1": f1_score(labels[val_idx], val_pred, average="macro"),
               "best_epoch": best_epoch, "epochs_trained": len(history),
               "device": str(device), "train_samples": len(train_idx),
               "n_val_samples": len(val_idx),
               "recall_by_class": recall_score(labels[val_idx], val_pred,
                                               labels=np.arange(6), average=None).tolist()}
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config = {"model": "LightweightTimeSeriesTransformer",
              "input_shape": [9, 128], "patch_size": 8, "embedding_dim": 64,
              "encoder_layers": 2, "attention_heads": 4, "feedforward_dim": 128,
              "dropout": 0.1, "preprocessing": "raw float32 signals; no extra standardization",
              "base_seed": args.seed, "fold_seed": fold_seed,
              "batch_size": args.batch_size, "max_epochs": args.epochs,
              "patience": args.patience, "optimizer": "AdamW",
              "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "pytorch_version": torch.__version__, "numpy_version": np.__version__,
              "split_file": str(SPLITS_FILE.relative_to(ROOT)),
              "split_sha256": hashlib.sha256(SPLITS_FILE.read_bytes()).hexdigest()}
    (fold_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {**metrics, "training_seconds": training_seconds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6),
                        help="Run one of the existing validation folds")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signals, labels, subjects = load_split("train")
    folds = load_folds(subjects)
    fold_numbers = [args.fold] if args.fold is not None else range(1, 6)
    for fold in fold_numbers:
        metrics = train_fold(fold, signals, labels, subjects, folds, args, device)
        print(json.dumps(metrics, indent=2), flush=True)

    metric_files = [OUTPUT_DIR / f"fold_{fold}" / "metrics.json" for fold in range(1, 6)]
    if all(path.is_file() for path in metric_files):
        keys = ("fold", "accuracy", "macro_f1", "best_epoch", "epochs_trained",
                "train_samples", "n_val_samples")
        table = pd.DataFrame([{key: json.loads(path.read_text(encoding="utf-8"))[key]
                               for key in keys} for path in metric_files])
        table.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
        summary = table[["accuracy", "macro_f1"]].agg(["mean", "std"])
        summary.to_csv(OUTPUT_DIR / "cv_summary.csv")
        print(summary, flush=True)


if __name__ == "__main__":
    main()
