from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score

root = Path.cwd()
data_dir = root / 'data/raw/UCI HAR Dataset/train'
out_dir = root / 'results'
X = pd.read_csv(data_dir / 'X_train.txt', sep=r'\s+', header=None).to_numpy()
y = np.loadtxt(data_dir / 'y_train.txt', dtype=int)
splits = pd.read_csv(out_dir / 'grrf_cv5_splits.csv')
selected = pd.read_csv(out_dir / 'grrf_cv5_selected_features.csv')
selection_times = pd.read_csv(out_dir / 'grrf_cv5_selection_times.csv')
assert X.shape == (7352, 561) and len(y) == len(splits) == 7352
assert (splits.row_index_python.to_numpy() == np.arange(7352)).all()
records = []
for fold in range(1, 6):
    tr = np.flatnonzero(splits.validation_fold.to_numpy() != fold)
    va = np.flatnonzero(splits.validation_fold.to_numpy() == fold)
    assert set(splits.subject.iloc[tr]).isdisjoint(set(splits.subject.iloc[va]))
    cases = [('Full_561', None, np.arange(561))]
    for g in sorted(selected.gamma_grrf.unique()):
        subset = selected.loc[(selected.fold == fold) & (np.isclose(selected.gamma_grrf, g)), 'feature_index_python'].to_numpy(dtype=int)
        assert len(subset) > 0 and len(np.unique(subset)) == len(subset)
        assert np.all((subset >= 0) & (subset < 561))
        cases.append((f'GRRF_gamma_{g:g}', float(g), subset))

    for name, gamma, cols in cases:
        model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
        start = perf_counter()
        model.fit(X[tr][:, cols], y[tr])
        training_sec = perf_counter() - start
        start = perf_counter()
        pred = model.predict(X[va][:, cols])
        prediction_sec = perf_counter() - start
        row = dict(fold=fold, method=name, gamma_grrf=gamma,
                   n_features=len(cols), n_val_samples=len(va),
                   accuracy=accuracy_score(y[va], pred),
                   macro_f1=f1_score(y[va], pred, average='macro'),
                   training_seconds=training_sec, prediction_seconds=prediction_sec)
        rec = recall_score(y[va], pred, labels=[1, 2, 3, 4, 5, 6], average=None, zero_division=0)
        for label, value in enumerate(rec, start=1):
            row[f'recall_class_{label}'] = value
        records.append(row)
        print(f'折 {fold}: {name:17s} 特征={len(cols):3d} Accuracy={row["accuracy"]:.4f} Macro-F1={row["macro_f1"]:.4f}')

fold_df = pd.DataFrame(records)
fold_df.to_csv(out_dir / 'grrf_cv5_fold_metrics.csv', index=False, encoding='utf-8-sig')
summary = fold_df.groupby('method', as_index=False).agg(
    n_features_mean=('n_features', 'mean'), n_features_std=('n_features', 'std'),
    accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
    macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
    training_seconds_mean=('training_seconds', 'mean'),
    prediction_seconds_mean=('prediction_seconds', 'mean'))
summary = summary.fillna(0)
summary.to_csv(out_dir / 'grrf_cv5_summary.csv', index=False, encoding='utf-8-sig')
print('\n===== 5 折平均结果（诊断，不是论文复现最终结果） =====')
print(summary.to_string(index=False))
print('\n输出: results/grrf_cv5_fold_metrics.csv 和 results/grrf_cv5_summary.csv')
print('GRRF特征选择时间另见 results/grrf_cv5_selection_times.csv')
