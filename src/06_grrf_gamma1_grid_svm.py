"""使用R生成的每折GRRF特征，进行SVM参数网格比较。
从项目根目录运行：.\\.venv\\Scripts\\python.exe .\\src\\06_grrf_gamma1_grid_svm.py
SVM参数是论文明确给出的 C=0.1、gamma=0.5；StandardScaler是本项目的明确实验选择，论文未完整说明。
不读取官方测试集。结果是用于选超参数的内部CV，不是无偏的最终泛化评估。
"""
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score

root = Path.cwd()
out = root / "results"
train_dir = root / "data" / "raw" / "UCI HAR Dataset" / "train"
split_file = out / "grrf_cv5_splits.csv"
feature_file = out / "grrf_gamma1_grid_selected_features.csv"
time_file = out / "grrf_gamma1_grid_selection_times.csv"
for file in [split_file, feature_file, time_file]:
    if not file.is_file():
        raise FileNotFoundError(f"缺少 {file}。请先完成第52步R参数搜索。")

X = pd.read_csv(train_dir / "X_train.txt", sep=r"\s+", header=None).to_numpy()
y = np.loadtxt(train_dir / "y_train.txt", dtype=int)
subjects = np.loadtxt(train_dir / "subject_train.txt", dtype=int)
splits = pd.read_csv(split_file)
features = pd.read_csv(feature_file)
times = pd.read_csv(time_file)
assert X.shape == (7352, 561)
assert len(y) == len(subjects) == len(splits) == 7352
assert np.array_equal(splits.row_index_python.to_numpy(), np.arange(7352))
assert np.array_equal(splits.subject.to_numpy(), subjects)
assert set(splits.validation_fold.unique()) == {1, 2, 3, 4, 5}
assert len(times) == 20 and not times.duplicated(["fold", "config_id"]).any()
assert times.config_id.nunique() == 4

records = []
fold_ids = splits.validation_fold.to_numpy(dtype=int)
for fold in range(1, 6):
    tr = np.flatnonzero(fold_ids != fold)
    va = np.flatnonzero(fold_ids == fold)
    assert set(subjects[tr]).isdisjoint(set(subjects[va]))
    configurations = [("Full_561", np.arange(561, dtype=int), None, None)]

    for cfg in sorted(times.config_id.unique()):
        sample = times[(times.fold == fold) & (times.config_id == cfg)]
        assert len(sample) == 1
        vals = features[(features.fold == fold) & (features.config_id == cfg)]
        cols = vals.feature_index_python.to_numpy(dtype=int)
        assert len(cols) == int(sample.n_features.iloc[0])
        assert len(np.unique(cols)) == len(cols)
        assert np.all((cols >= 0) & (cols < 561))
        configurations.append((cfg, cols, int(sample.ntree.iloc[0]), int(sample.mtry.iloc[0])))

    for cfg, cols, ntree, mtry in configurations:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=0.1, gamma=0.5))
        t0 = perf_counter()
        model.fit(X[tr][:, cols], y[tr])
        train_s = perf_counter() - t0
        t0 = perf_counter()
        pred = model.predict(X[va][:, cols])
        pred_s = perf_counter() - t0
        recalls = recall_score(y[va], pred, labels=[1, 2, 3, 4, 5, 6], average=None, zero_division=0)
        rec = dict(fold=fold, config_id=cfg, method="Full" if cfg == "Full_561" else "GRRF",
                   n_features=len(cols), ntree=ntree, mtry=mtry,
                   svm_C=0.1, svm_gamma=0.5, scaler="StandardScaler",
                   accuracy=accuracy_score(y[va], pred),
                   macro_f1=f1_score(y[va], pred, labels=[1, 2, 3, 4, 5, 6], average="macro", zero_division=0),
                   training_seconds=train_s, prediction_seconds=pred_s, n_validation_samples=len(va))
        for label, value in enumerate(recalls, start=1):
            rec[f"recall_class_{label}"] = float(value)
        records.append(rec)
        print(f"第{fold}折 {cfg:12s} 特征={len(cols):3d} Accuracy={rec['accuracy']:.4f} Macro-F1={rec['macro_f1']:.4f}")

fold_df = pd.DataFrame(records)
assert len(fold_df) == 25
fold_df.to_csv(out / "grrf_gamma1_grid_fold_metrics.csv", index=False, encoding="utf-8-sig")
summary = fold_df.groupby("config_id", as_index=False).agg(
    n_features_mean=("n_features", "mean"),
    n_features_std=("n_features", "std"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
    training_seconds_mean=("training_seconds", "mean"),
    prediction_seconds_mean=("prediction_seconds", "mean"),
)
sel_summary = times.groupby("config_id", as_index=False).agg(
    selection_seconds_mean=("selection_total_seconds", "mean")
)
summary = summary.merge(sel_summary, on="config_id", how="left", validate="one_to_one")
summary["selection_seconds_mean"] = summary.selection_seconds_mean.fillna(0)
summary["total_seconds_mean"] = (summary.selection_seconds_mean + summary.training_seconds_mean
                                 + summary.prediction_seconds_mean)
summary = summary.sort_values("config_id")
summary.to_csv(out / "grrf_gamma1_grid_summary.csv", index=False, encoding="utf-8-sig")

# 配对比较：同一验证折下的GRRF与561维Full SVM(论文SVM参数)
base = fold_df[fold_df.config_id == "Full_561"][["fold", "accuracy", "macro_f1"]].rename(
    columns={"accuracy": "full_accuracy", "macro_f1": "full_macro_f1"})
paired = fold_df[fold_df.method == "GRRF"].merge(base, on="fold", validate="many_to_one")
paired["accuracy_drop_pp"] = (paired.full_accuracy - paired.accuracy) * 100
paired["macro_f1_drop_pp"] = (paired.full_macro_f1 - paired.macro_f1) * 100
paired["reduction_percent"] = (1 - paired.n_features / 561) * 100
paired.to_csv(out / "grrf_gamma1_grid_paired.csv", index=False, encoding="utf-8-sig")

# 这是模型选择：最高平均Macro-F1；仅作为调参结论，不能作为独立的最终泛化成绩。
grrf_summary = summary[summary.config_id != "Full_561"].copy()
best = grrf_summary.sort_values(["macro_f1_mean", "accuracy_mean", "n_features_mean"],
                                ascending=[False, False, True]).iloc[0]
selection = pd.DataFrame([{
    "chosen_config_id": best.config_id,
    "criterion": "highest 5-fold mean macro_f1, then accuracy, then fewer features",
    "gamma_grrf": 1.0,
    "ntree": int(times.loc[times.config_id == best.config_id, "ntree"].iloc[0]),
    "mtry": int(times.loc[times.config_id == best.config_id, "mtry"].iloc[0]),
    "mean_cv_macro_f1": best.macro_f1_mean,
    "note": "selected on training-set inner GroupKFold; NOT independent test performance",
}])
selection.to_csv(out / "grrf_gamma1_grid_choice.csv", index=False, encoding="utf-8-sig")
print("\n===== 5折内部参数搜索汇总（论文SVM参数，项目标准化） =====")
print(summary.round(4).to_string(index=False))
print("\n当前训练集内部最高平均Macro-F1的GRRF组合：", best.config_id)
print("注意：尚未运行任何官方测试集评估；不能直接与论文99.30%当作同条件对比。")
print("\n生成: grrf_gamma1_grid_fold_metrics.csv / summary.csv / paired.csv / choice.csv")
