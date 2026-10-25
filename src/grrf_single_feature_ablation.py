

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. 项目路径
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

# 3. 读取之前固定的用户分组
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

assert train_subjects.isdisjoint(val_subjects)
assert len(train_idx) + len(val_idx) == len(X)

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# 4. 读取GRRF实际选中的特征
selected_df = pd.read_csv(
    results_dir / "grrf_group_selected_features.csv"
)

def get_features(gamma):
    group = selected_df[
        selected_df["gamma_grrf"] == gamma
    ]
    return sorted(
        group["feature_index_python"].astype(int).tolist()
    )

features_075 = get_features(0.75)
features_1 = get_features(1.0)

# 自动求共同特征和各自独有的特征
common_features = sorted(
    set(features_075) & set(features_1)
)

only_075 = sorted(set(features_075) - set(features_1))
only_1 = sorted(set(features_1) - set(features_075))

assert len(features_075) == 7
assert len(features_1) == 7
assert len(common_features) == 6
assert len(only_075) == 1
assert len(only_1) == 1

print("共同特征：", common_features)
print("仅gamma=0.75选择：", only_075)
print("仅gamma=1选择：", only_1)

# 5. 设计控制变量实验
experiments = {
    "Full_561": list(range(561)),
    "Common_6": common_features,
    "Common_6_plus_gamma075_feature":
        common_features + only_075,
    "Common_6_plus_gamma1_feature":
        common_features + only_1,
}

# 6. 使用相同数据、相同SVM参数进行评价
records = []
reports = []
confusion_rows = []

for name, cols in experiments.items():
    cols = sorted(cols)

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )
    )

    start = time.perf_counter()
    model.fit(X_train[:, cols], y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred = model.predict(X_val[:, cols])
    prediction_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_val, pred)
    macro_f1 = f1_score(
        y_val, pred, average="macro"
    )

    records.append({
        "method": name,
        "n_features": len(cols),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "training_seconds": train_seconds,
        "prediction_seconds": prediction_seconds
    })

    # 保存各类别指标
    report = classification_report(
        y_val,
        pred,
        labels=[1, 2, 3, 4, 5, 6],
        output_dict=True,
        zero_division=0
    )

    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue

        reports.append({
            "method": name,
            "class_label": label,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1-score"],
            "support": metrics["support"]
        })

    # 保存混淆矩阵
    cm = confusion_matrix(
        y_val, pred, labels=[1, 2, 3, 4, 5, 6]
    )

    for i in range(6):
        for j in range(6):
            confusion_rows.append({
                "method": name,
                "true_class": i + 1,
                "predicted_class": j + 1,
                "count": int(cm[i, j])
            })

    print(
        f"{name}: features={len(cols)}, "
        f"Accuracy={accuracy:.4f}, "
        f"Macro-F1={macro_f1:.4f}"
    )

# 7. 保存所有实验结果
pd.DataFrame(records).to_csv(
    results_dir / "grrf_single_feature_ablation.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(reports).to_csv(
    results_dir / "grrf_single_feature_class_report.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(confusion_rows).to_csv(
    results_dir / "grrf_single_feature_confusion.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n控制变量实验完成。")
print(pd.DataFrame(records).to_string(index=False))
print("\n结果保存在 results 文件夹中。")
