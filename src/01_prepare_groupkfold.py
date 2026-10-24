from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

root = Path.cwd()
subject_path = root / 'data/raw/UCI HAR Dataset/train/subject_train.txt'
if not subject_path.is_file():
    raise FileNotFoundError(f'请先在项目根目录运行本脚本，缺少: {subject_path}')
subjects = np.loadtxt(subject_path, dtype=int)
assert len(subjects) == 7352
folds = np.zeros(len(subjects), dtype=int)
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(np.zeros(len(subjects)), groups=subjects), start=1):
    assert set(subjects[train_idx]).isdisjoint(set(subjects[val_idx]))
    folds[val_idx] = fold
    print(f'第{fold}折: 训练 {len(train_idx)} 条, 验证 {len(val_idx)} 条, 验证用户 {sorted(set(subjects[val_idx]))}')
assert np.all((folds >= 1) & (folds <= 5))
assert len(set(subjects)) >= 5
assert pd.DataFrame({'subject': subjects, 'fold': folds}).groupby('subject').fold.nunique().eq(1).all()
out = root / 'results'
out.mkdir(exist_ok=True)
p = out / 'grrf_cv5_splits.csv'
pd.DataFrame({'row_index_python': np.arange(len(subjects)), 'subject': subjects, 'validation_fold': folds}).to_csv(p, index=False)
print('保存成功:', p)
