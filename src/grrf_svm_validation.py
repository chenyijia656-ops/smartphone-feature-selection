
# GRRF特征选择后的SVM验证实验
# 只使用官方训练集中的训练/验证用户
# 不使用官方测试集

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# 1. 项目目录
root = Path.cwd()
data_dir = root / "data" / "raw" / "UCI HAR Dataset"
results_dir = root / "results"

# 2. 读取官方训练集
X = pd.read_csv(
    data_dir / "train" / "X_train.txt",
    sep=r"\s+",
    header=None
).to_numpy()

y = pd.read_csv(
    data_dir / "train" / "y_train.txt",
    sep=r"\s+",
    header=None
).iloc[:, 0].to_numpy()

assert X.shape == (7352, 561)
assert len(y) == 7352

# 3. 读取R生成的分组结果
split_df = pd.read_csv(
    results_dir / "grrf_group_split.csv"
)

train_idx = split_df.loc[
    split_df["split"] == "train",
    "row_index_python"
].to_numpy(dtype=int)

val_idx = split_df.loc[
    split_df["split"] == "validation",
    "row_index_python"
].to_numpy(dtype=int)

train_subjects = set(
    split_df.loc[
        split_df["split"] == "train", "subject"
    ]
)

val_subjects = set(
    split_df.loc[
        split_df["split"] == "validation", "subject"
    ]
)

assert len(train_subjects & val_subjects) == 0
assert len(train_idx) + len(val_idx) == 7352

# 4. 读取GRRF选中的特征
selected_df = pd.read_csv(
    results_dir / "grrf_group_selected_features.csv"
)

# 5. 建立比较方案
experiments = [
    ("Full_561", np.arange(561))
]

for gamma_value, group in selected_df.groupby("gamma_grrf"):
    feature_indices = (
        group["feature_index_python"]
        .to_numpy(dtype=int)
    )

    experiments.append((
        f"GRRF_gamma_{gamma_value:g}",
        feature_indices
    ))

# 6. 逐个训练SVM
records = []

for method, cols in experiments:

    X_train = X[train_idx][:, cols]
    X_val = X[val_idx][:, cols]

    y_train = y[train_idx]
    y_val = y[val_idx]

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )
    )

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X_val)
    predict_time = time.perf_counter() - predict_start

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(
        y_val, y_pred, average="macro"
    )

    records.append({
        "method": method,
        "n_features": len(cols),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "training_seconds": train_time,
        "prediction_seconds": predict_time
    })

    print(
        f"{method}: "
        f"特征数={len(cols)}, "
        f"Accuracy={acc:.4f}, "
        f"Macro-F1={macro_f1:.4f}"
    )

# 7. 保存结果
result_df = pd.DataFrame(records)

output_path = (
    results_dir / "grrf_svm_group_validation.csv"
)

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 实验汇总 =====")
print(result_df.to_string(index=False))
print("\n结果已保存到：", output_path)
