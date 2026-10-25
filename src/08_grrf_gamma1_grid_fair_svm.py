r"""第54步：复用既有 GRRF gamma=1 的5折特征，公平比较 SVM。

从项目根目录运行：
    & ".\.venv\Scripts\python.exe" ".\src\08_grrf_gamma1_grid_fair_svm.py"

不重新运行 GRRF；只读取官方 TRAIN 的数据，不读取官方 TEST。
所有 Full 与 GRRF 实验使用 StandardScaler + RBF-SVC(C=1.0, gamma='scale')。
注意：这些折用于调参；平均交叉验证成绩不是独立的最终测试成绩。
"""
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

root = Path.cwd()
train_dir = root / "data" / "raw" / "UCI HAR Dataset" / "train"
out = root / "results"
required_files = {
    "训练特征": train_dir / "X_train.txt",
    "训练标签": train_dir / "y_train.txt",
    "参与者": train_dir / "subject_train.txt",
    "五折划分": out / "grrf_cv5_splits.csv",
    "特征索引": out / "grrf_gamma1_grid_selected_features.csv",
    "特征选择时间": out / "grrf_gamma1_grid_selection_times.csv",
}
for name, filepath in required_files.items():
    if not filepath.is_file():
        raise FileNotFoundError(f"缺少{name}: {filepath}")

X = pd.read_csv(required_files["训练特征"], sep=r"\s+", header=None).to_numpy()
y = np.loadtxt(required_files["训练标签"], dtype=int)
subjects = np.loadtxt(required_files["参与者"], dtype=int)
splits = pd.read_csv(required_files["五折划分"])
selected = pd.read_csv(required_files["特征索引"])
times = pd.read_csv(required_files["特征选择时间"])

assert X.shape == (7352, 561), X.shape
assert len(y) == len(subjects) == 7352
for name, df, cols in [
    ("五折划分", splits, {"row_index_python", "subject", "validation_fold"}),
    ("特征索引", selected, {"fold", "config_id", "feature_index_python"}),
    ("特征选择时间", times, {"fold", "config_id", "ntree", "mtry", "n_features", "selection_total_seconds"}),
]:
    if not cols.issubset(df.columns):
        raise ValueError(f"{name}列不匹配，缺少：{sorted(cols - set(df.columns))}")

splits = splits.sort_values("row_index_python").reset_index(drop=True)
assert np.array_equal(splits.row_index_python.to_numpy(), np.arange(7352))
assert np.array_equal(splits.subject.to_numpy(), subjects)
assert sorted(splits.validation_fold.unique().tolist()) == [1, 2, 3, 4, 5]
assert (splits.groupby("subject").validation_fold.nunique() == 1).all()
assert not times.duplicated(["fold", "config_id"]).any()
assert len(times) == 20, f"参数-折组合应有20条记录，实际{len(times)}"
config_ids = sorted(times.config_id.unique())
assert len(config_ids) == 4
assert not selected.empty

fold_ids = splits.validation_fold.to_numpy(dtype=int)
rows = []
for fold in range(1, 6):
    tr = np.flatnonzero(fold_ids != fold)
    va = np.flatnonzero(fold_ids == fold)
    assert set(subjects[tr]).isdisjoint(set(subjects[va]))
    configs = [("Full_561", np.arange(561, dtype=int), 0.0)]
    for config_id in config_ids:
        detail = times[(times.fold == fold) & (times.config_id == config_id)]
        picked = selected[(selected.fold == fold) & (selected.config_id == config_id)]
        assert len(detail) == 1, (fold, config_id)
        indices = picked.feature_index_python.to_numpy(dtype=int)
        assert len(indices) == int(detail.n_features.iloc[0]), (fold, config_id)
        assert len(np.unique(indices)) == len(indices)
        assert np.all((indices >= 0) & (indices < 561))
        configs.append((config_id, indices, float(detail.selection_total_seconds.iloc[0])))

    for config_id, indices, selection_s in configs:
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale"),
        )
        start = perf_counter()
        model.fit(X[tr][:, indices], y[tr])
        training_s = perf_counter() - start
        start = perf_counter()
        pred = model.predict(X[va][:, indices])
        prediction_s = perf_counter() - start
        recalls = recall_score(
            y[va], pred, labels=[1, 2, 3, 4, 5, 6], average=None, zero_division=0
        )
        record = {
            "fold": fold,
            "config_id": config_id,
            "n_features": len(indices),
            "svm_C": 1.0,
            "svm_gamma": "scale",
            "scaler": "StandardScaler",
            "accuracy": accuracy_score(y[va], pred),
            "macro_f1": f1_score(
                y[va], pred, labels=[1, 2, 3, 4, 5, 6], average="macro", zero_division=0
            ),
            "selection_seconds": selection_s,
            "training_seconds": training_s,
            "prediction_seconds": prediction_s,
            "n_validation_samples": len(va),
        }
        for label, value in enumerate(recalls, start=1):
            record[f"recall_class_{label}"] = float(value)
        rows.append(record)
        print(
            f"第{fold}折 {config_id:11s} 特征={len(indices):3d} "
            f"Accuracy={record['accuracy']:.4f} Macro-F1={record['macro_f1']:.4f}"
        )

fold_df = pd.DataFrame(rows)
assert len(fold_df) == 25
summary = fold_df.groupby("config_id", as_index=False).agg(
    n_features_mean=("n_features", "mean"),
    n_features_std=("n_features", "std"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
    selection_seconds_mean=("selection_seconds", "mean"),
    training_seconds_mean=("training_seconds", "mean"),
    prediction_seconds_mean=("prediction_seconds", "mean"),
)
summary["total_seconds_mean"] = (
    summary["selection_seconds_mean"] + summary["training_seconds_mean"] + summary["prediction_seconds_mean"]
)
base = fold_df[fold_df.config_id == "Full_561"]["fold accuracy macro_f1".split()].rename(
    columns={"accuracy": "full_accuracy", "macro_f1": "full_macro_f1"}
)
paired = fold_df[fold_df.config_id != "Full_561"].merge(base, on="fold", validate="many_to_one")
paired["accuracy_drop_pp"] = (paired.full_accuracy - paired.accuracy) * 100
paired["macro_f1_drop_pp"] = (paired.full_macro_f1 - paired.macro_f1) * 100
paired["reduction_percent"] = (1 - paired.n_features / 561) * 100

prefix = "grrf_gamma1_grid_fair_svm"
fold_df.to_csv(out / f"{prefix}_fold_metrics.csv", index=False, encoding="utf-8-sig")
summary.to_csv(out / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
paired.to_csv(out / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")

print("\n===== 统一 SVM 协议的 5 折汇总：C=1.0, gamma=scale =====")
print(summary.round(4).to_string(index=False))
print("\n文件已保存到 results/：")
print(f"  {prefix}_fold_metrics.csv")
print(f"  {prefix}_summary.csv")
print(f"  {prefix}_paired.csv")
print("注意：这是训练集内部调参比较，不是独立的最终测试结果。")
