
# GRRF 按用户分组验证
# 初步验证实验，不是最终论文10折复现

library(RRF)

set.seed(42)

root <- normalizePath(getwd(), winslash = "/")
data_dir <- file.path(root, "data", "raw", "UCI HAR Dataset")
out_dir <- file.path(root, "results")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 1. 读取官方训练集与参与者编号
X <- as.matrix(read.table(
  file.path(data_dir, "train", "X_train.txt"),
  header = FALSE
))

y <- factor(scan(
  file.path(data_dir, "train", "y_train.txt"),
  quiet = TRUE
))

subjects <- scan(
  file.path(data_dir, "train", "subject_train.txt"),
  quiet = TRUE
)

stopifnot(nrow(X) == 7352)
stopifnot(ncol(X) == 561)
stopifnot(length(y) == 7352)
stopifnot(length(subjects) == 7352)

# 2. 按参与者划分训练和验证数据
unique_subjects <- sort(unique(subjects))

set.seed(2026)
n_train_subjects <- floor(0.75 * length(unique_subjects))

train_subjects <- sort(sample(
  unique_subjects,
  size = n_train_subjects
))

train_rows <- which(subjects %in% train_subjects)
val_rows <- which(!(subjects %in% train_subjects))

stopifnot(length(intersect(
  unique(subjects[train_rows]),
  unique(subjects[val_rows])
)) == 0)

cat("训练用户数：", length(unique(subjects[train_rows])), "\n")
cat("验证用户数：", length(unique(subjects[val_rows])), "\n")
cat("训练样本数：", length(train_rows), "\n")
cat("验证样本数：", length(val_rows), "\n")

# 保存划分，Python 后续使用相同数据
split_df <- data.frame(
  row_index_python = seq_along(subjects) - 1L,
  subject = subjects,
  split = ifelse(
    seq_along(subjects) %in% train_rows,
    "train",
    "validation"
  )
)

write.csv(
  split_df,
  file.path(out_dir, "grrf_group_split.csv"),
  row.names = FALSE
)

# 3. 只使用训练用户的数据
X_inner <- X[train_rows, , drop = FALSE]
y_inner <- y[train_rows]

ntree <- 100
mtry <- floor(sqrt(ncol(X_inner)))
gamma_values <- c(0, 0.25, 0.5, 0.75, 1)

# 4. 训练普通RF，得到训练用户上的重要性
set.seed(42)

rf <- RRF(
  x = X_inner,
  y = y_inner,
  ntree = ntree,
  mtry = mtry,
  flagReg = 0,
  importance = TRUE
)

importance_rf <- rf$importance[, "MeanDecreaseGini"]
importance_rf[!is.finite(importance_rf)] <- 0

stopifnot(max(importance_rf) > 0)

imp <- importance_rf / max(importance_rf)

# 5. 在相同训练用户上运行不同gamma的GRRF
selected_rows <- list()
summary_rows <- list()

for (g in gamma_values) {

  coef_reg <- (1 - g) + g * imp

  set.seed(42)

  start_time <- proc.time()[3]

  grrf <- RRF(
    x = X_inner,
    y = y_inner,
    ntree = ntree,
    mtry = mtry,
    coefReg = coef_reg,
    flagReg = 1
  )

  elapsed <- unname(proc.time()[3] - start_time)

  selected_r <- sort(unique(as.integer(grrf$feaSet)))

  stopifnot(length(selected_r) > 0)
  stopifnot(all(selected_r >= 1 & selected_r <= 561))

  key <- as.character(g)

  selected_rows[[key]] <- data.frame(
    gamma_grrf = g,
    feature_index_python = selected_r - 1L
  )

  summary_rows[[key]] <- data.frame(
    gamma_grrf = g,
    ntree = ntree,
    mtry = mtry,
    selected_features = length(selected_r),
    selection_time_seconds = elapsed
  )

  cat(
    "gamma =", g,
    "；选中特征数 =", length(selected_r),
    "；GRRF耗时 =", round(elapsed, 3), "秒\n"
  )
}

# 6. 保存结果
selected_df <- do.call(rbind, selected_rows)
summary_df <- do.call(rbind, summary_rows)

write.csv(
  selected_df,
  file.path(out_dir, "grrf_group_selected_features.csv"),
  row.names = FALSE
)

write.csv(
  summary_df,
  file.path(out_dir, "grrf_group_selection_summary.csv"),
  row.names = FALSE
)

cat("\n按用户分组的GRRF实验完成。\n")
