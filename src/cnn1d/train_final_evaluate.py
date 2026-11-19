"""Train the preselected CNN on all official training data, then test once."""

import csv
import hashlib
import json
import time

from train import SmallCNN1D, evaluate, make_loader, set_seed
from data import ROOT, load_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn


OUTPUT_DIR = ROOT / "results" / "cnn1d" / "official_test"
SPLITS_FILE = ROOT / "results" / "grrf_cv5_splits.csv"
SEED = 42
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
CLASS_NAMES = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING")


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, _ = load_split("train")
    x_test, y_test, test_subjects = load_split("test")
    assert x_train.shape == (7352, 9, 128) and x_test.shape == (2947, 9, 128)

    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True, seed=SEED)
    test_loader = make_loader(x_test, y_test, BATCH_SIZE, seed=SEED)
    model = SmallCNN1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = []

    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        train_loss = loss_sum / len(y_train)
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"epoch {epoch:02d}: train_loss={train_loss:.6f}")
    training_seconds = time.perf_counter() - start

    # The official test set is evaluated only after the fixed training schedule.
    predictions, test_loss = evaluate(model, test_loader, device, criterion)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=np.arange(6), zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=np.arange(6))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "test_loss": float(test_loss),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "epochs": EPOCHS,
        "device": str(device),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    with (OUTPUT_DIR / "train_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_loss"))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["epoch"] for row in history],
            [row["train_loss"] for row in history], marker="o")
    ax.set(title="Final model training loss", xlabel="Epoch", ylabel="Cross-entropy loss",
           xticks=range(1, EPOCHS + 1))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png", dpi=160)
    plt.close(fig)

    pd.DataFrame({"class_id": np.arange(6), "activity": CLASS_NAMES,
                  "precision": precision, "recall": recall,
                  "f1": f1, "support": support}).to_csv(
                      OUTPUT_DIR / "class_metrics.csv", index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv", index_label="true_activity")
    pd.DataFrame({"row_index_python": np.arange(len(y_test)),
                  "subject": test_subjects, "true_label": y_test,
                  "predicted_label": predictions}).to_csv(
                      OUTPUT_DIR / "predictions.csv", index=False)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config = {
        "model": "SmallCNN1D", "input_shape": [9, 128],
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "loss": "CrossEntropyLoss", "data_source": "UCI HAR official train/test",
        "epoch_selection": "highest mean five-fold validation Macro-F1 at a common epoch",
        "split_file_for_epoch_selection": str(SPLITS_FILE.relative_to(ROOT)),
        "split_sha256": hashlib.sha256(SPLITS_FILE.read_bytes()).hexdigest(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "pytorch_version": torch.__version__, "numpy_version": np.__version__,
        "device": str(device),
    }
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"training_seconds={training_seconds:.3f}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
