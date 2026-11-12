"""Evaluate a compact 1D-CNN using the project's existing five folds."""

import argparse
import csv
import hashlib
import json
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import ROOT, load_split


OUTPUT_DIR = ROOT / "results" / "cnn1d"
SPLITS_FILE = ROOT / "results" / "grrf_cv5_splits.csv"


class SmallCNN1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(9, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[np.ndarray, float]:
    model.eval()
    predictions = []
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            total_loss += criterion(logits, yb.to(device)).item() * len(yb)
            count += len(yb)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions), total_loss / count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_curve(history: list[dict], best_epoch: int, path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    for ax in axes:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_folds(subjects: np.ndarray) -> np.ndarray:
    table = pd.read_csv(SPLITS_FILE)
    required = {"row_index_python", "subject", "validation_fold"}
    if not required.issubset(table.columns) or len(table) != len(subjects):
        raise ValueError("Five-fold split file has missing columns or sample count mismatch")
    if not np.array_equal(table.row_index_python.to_numpy(), np.arange(len(subjects))):
        raise ValueError("Five-fold split rows do not match training data order")
    if not np.array_equal(table.subject.to_numpy(), subjects):
        raise ValueError("Five-fold split subjects do not match training data")
    folds = table.validation_fold.to_numpy(dtype=int)
    if set(folds) != set(range(1, 6)):
        raise ValueError("Expected validation folds 1..5")
    if table.groupby("subject").validation_fold.nunique().ne(1).any():
        raise ValueError("A subject appears in multiple validation folds")
    return folds


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = False, seed: int = 0) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def train_fold(fold: int, x_train: np.ndarray, y_train: np.ndarray,
               subjects: np.ndarray, folds: np.ndarray,
               args: argparse.Namespace, device: torch.device) -> dict:
    train_idx = np.flatnonzero(folds != fold)
    val_idx = np.flatnonzero(folds == fold)
    assert set(subjects[train_idx]).isdisjoint(set(subjects[val_idx]))
    fold_seed = args.seed + fold
    set_seed(fold_seed)
    train_loader = make_loader(x_train[train_idx], y_train[train_idx],
                               args.batch_size, True, fold_seed)
    val_loader = make_loader(x_train[val_idx], y_train[val_idx],
                             args.batch_size, seed=fold_seed)
    model = SmallCNN1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_f1, stale, best_state = -1.0, 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        val_pred, val_loss = evaluate(model, val_loader, device, criterion)
        val_f1 = f1_score(y_train[val_idx], val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_idx),
                        "val_loss": val_loss,
                        "val_accuracy": accuracy_score(y_train[val_idx], val_pred),
                        "val_macro_f1": val_f1})
        print(f"fold {fold} epoch {epoch:02d}: loss={history[-1]['train_loss']:.4f}, val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    val_pred, _ = evaluate(model, val_loader, device, criterion)
    training_seconds = time.perf_counter() - start
    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, fold_dir / "model.pt")
    with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    best_epoch = int(np.argmax([row["val_macro_f1"] for row in history]) + 1)
    save_curve(history, best_epoch, fold_dir / "training_curves.png")
    run_config = {"base_seed": args.seed, "fold_seed": fold_seed,
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "cudnn_benchmark": torch.backends.cudnn.benchmark,
                  "cudnn_deterministic": torch.backends.cudnn.deterministic,
                  "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                  "pytorch_version": torch.__version__, "numpy_version": np.__version__,
                  "batch_size": args.batch_size, "max_epochs": args.epochs,
                  "patience": args.patience, "learning_rate": 1e-3,
                  "split_file": str(SPLITS_FILE.relative_to(ROOT)),
                  "split_sha256": hashlib.sha256(SPLITS_FILE.read_bytes()).hexdigest()}
    (fold_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    metrics = {"fold": fold, "accuracy": accuracy_score(y_train[val_idx], val_pred),
               "macro_f1": f1_score(y_train[val_idx], val_pred, average="macro"),
               "best_epoch": best_epoch,
               "epochs_trained": len(history),
               "training_seconds": training_seconds, "device": str(device),
               "train_samples": len(train_idx), "n_val_samples": len(val_idx),
               "recall_by_class": recall_score(y_train[val_idx], val_pred,
                                               labels=np.arange(6), average=None).tolist()}
    cm = confusion_matrix(y_train[val_idx], val_pred, labels=np.arange(6))
    np.savetxt(fold_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    saved_metrics = {key: value for key, value in metrics.items() if key != "training_seconds"}
    (fold_dir / "metrics.json").write_text(json.dumps(saved_metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=range(1, 6),
                        help="Run only this existing validation fold (1-5)")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, subjects = load_split("train")
    folds = load_folds(subjects)
    fold_numbers = [args.fold] if args.fold is not None else range(1, 6)
    rows = [train_fold(fold, x_train, y_train, subjects, folds, args, device)
            for fold in fold_numbers]
    if args.fold is not None:
        print(json.dumps(rows[0], indent=2))
    metric_files = [OUTPUT_DIR / f"fold_{fold}" / "metrics.json" for fold in range(1, 6)]
    if all(path.is_file() for path in metric_files):
        keys = ("fold", "accuracy", "macro_f1", "best_epoch", "epochs_trained",
                "train_samples", "n_val_samples")
        table = pd.DataFrame([{key: json.loads(path.read_text(encoding="utf-8"))[key]
                               for key in keys} for path in metric_files])
        table.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
        summary = table[["accuracy", "macro_f1"]].agg(["mean", "std"])
        summary.to_csv(OUTPUT_DIR / "cv_summary.csv")
        print(summary)


if __name__ == "__main__":
    main()
