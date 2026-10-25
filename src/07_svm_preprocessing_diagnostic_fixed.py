from pathlib import Path
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "raw" / "UCI HAR Dataset"
RESULTS_DIR = ROOT / "results"

X = pd.read_csv(
    DATA_DIR / "train" / "X_train.txt",
    sep=r"\s+",
    header=None
).to_numpy()

y = pd.read_csv(
    DATA_DIR / "train" / "y_train.txt",
    sep=r"\s+",
    header=None
).iloc[:, 0].to_numpy()

splits_path = RESULTS_DIR / "grrf_cv5_splits.csv"
splits = pd.read_csv(splits_path)

required = {"row_index_python", "subject", "validation_fold"}
missing = required - set(splits.columns)
if missing:
    raise ValueError(
        f"{splits_path.name} 缺少列 {sorted(missing)}；"
        f"实际列名为：{list(splits.columns)}"
    )

# 划分文件是一行一条样本，validation_fold 表示该样本在第几折作为验证样本。
# 其他折的样本则用于训练，绝不能先在全训练集上筛选特征。
splits = splits.sort_values("row_index_python").reset_index(drop=True)
indices = splits["row_index_python"].to_numpy(dtype=int)
if len(indices) != len(X) or not (indices == range(len(X))).all():
    raise ValueError("划分文件的索引不完整、重复或不对应官方训练集")

actual_subjects = pd.read_csv(
    DATA_DIR / "train" / "subject_train.txt", sep=r"\s+", header=None
).iloc[:, 0].to_numpy(dtype=int)
if not (splits["subject"].to_numpy(dtype=int) == actual_subjects).all():
    raise ValueError("划分文件中subject顺序与官方subject_train.txt不一致")

if splits["validation_fold"].isna().any():
    raise ValueError("有样本未分配验证折")
if (splits.groupby("subject")["validation_fold"].nunique() != 1).any():
    raise ValueError("存在同一个用户被分到多个验证折的情况")

folds = sorted(splits["validation_fold"].astype(int).unique())
if folds != [1, 2, 3, 4, 5]:
    raise ValueError(f"预期验证折为1到5，实际为{folds}")
print("已核验划分文件：5折，每个参与者只属于一个验证折。")

configs = {
    "paper_C0.1_gamma0.5_no_scaler": {
        "scaler": False, "C": 0.1, "gamma": 0.5
    },
    "paper_C0.1_gamma0.5_with_scaler": {
        "scaler": True, "C": 0.1, "gamma": 0.5
    },
    "project_C1_gamma_scale_with_scaler": {
        "scaler": True, "C": 1.0, "gamma": "scale"
    },
    "C0.1_gamma_scale_with_scaler": {
        "scaler": True, "C": 0.1, "gamma": "scale"
    },
}

rows = []
for fold in folds:
    # validation_fold==fold：本折验证样本；其余样本：本折训练样本
    train_idx = splits.loc[
        splits["validation_fold"] != fold,
        "row_index_python"
    ].to_numpy(dtype=int)
    val_idx = splits.loc[
        splits["validation_fold"] == fold,
        "row_index_python"
    ].to_numpy(dtype=int)

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"第{fold}折没有训练样本或验证样本")
    if set(train_idx) & set(val_idx):
        raise ValueError(f"第{fold}折训练/验证样本索引重叠")
    train_subjects = set(actual_subjects[train_idx])
    val_subjects = set(actual_subjects[val_idx])
    if not train_subjects.isdisjoint(val_subjects):
        raise ValueError(f"第{fold}折训练/验证参与者重叠")
    print(f"第{fold}折：训练{len(train_idx)}条，验证{len(val_idx)}条；用户无重叠")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    for name, cfg in configs.items():
        if cfg["scaler"]:
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"])
            )
        else:
            model = SVC(
                kernel="rbf", C=cfg["C"], gamma=cfg["gamma"]
            )

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        row = {
            "fold": int(fold),
            "config": name,
            "n_features": X.shape[1],
            "accuracy": accuracy_score(y_val, pred),
            "macro_f1": f1_score(
                y_val, pred, average="macro", zero_division=0
            ),
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
        }
        rows.append(row)
        print(
            f"折{fold} {name}: "
            f"Accuracy={row['accuracy']:.4f}, "
            f"Macro-F1={row['macro_f1']:.4f}"
        )

fold_df = pd.DataFrame(rows)
summary = (
    fold_df.groupby("config")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    .reset_index()
)

fold_path = RESULTS_DIR / "svm_preprocessing_diagnostic_folds.csv"
summary_path = RESULTS_DIR / "svm_preprocessing_diagnostic_summary.csv"
fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\n===== 全特征 SVM 预处理/参数诊断汇总 =====")
print(summary.to_string(index=False))
print("\n已保存：", fold_path)
print("已保存：", summary_path)
