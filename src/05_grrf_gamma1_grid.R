# 第52步：GRRF gamma=1 参数网格（训练集内部5折按用户分组）
# 运行：Rscript src/05_grrf_gamma1_grid.R --smoke
# 完整：Rscript src/05_grrf_gamma1_grid.R
# 正式结果不使用官方测试集。脚本会把每一折缓存为RDS，重复执行可从已完成的折继续。
suppressPackageStartupMessages(library(RRF))

root <- normalizePath(getwd(), winslash = "/")
data_dir <- file.path(root, "data", "raw", "UCI HAR Dataset", "train")
out_dir <- file.path(root, "results")
split_path <- file.path(out_dir, "grrf_cv5_splits.csv")
if (!file.exists(split_path)) stop("找不到 grrf_cv5_splits.csv：请先完成第50步，并从项目根目录运行")

X <- as.matrix(read.table(file.path(data_dir, "X_train.txt"), header = FALSE))
y <- factor(scan(file.path(data_dir, "y_train.txt"), quiet = TRUE))
subjects <- scan(file.path(data_dir, "subject_train.txt"), quiet = TRUE)
splits <- read.csv(split_path)
stopifnot(nrow(X) == 7352L, ncol(X) == 561L, length(y) == 7352L)
stopifnot(nrow(splits) == 7352L, identical(as.integer(splits$row_index_python), 0:7351))
stopifnot(all(splits$subject == subjects), setequal(splits$validation_fold, 1:5))
stopifnot(all(tapply(splits$validation_fold, splits$subject, function(z) length(unique(z))) == 1))

# gamma 固定为论文写明的 1，候选 ntree/mtry 是我们的探索性网格，并非论文提供的原始网格。
gamma_grrf <- 1.0
configs <- expand.grid(ntree = c(100L, 500L), mtry = c(23L, 56L))
configs <- configs[order(configs$ntree, configs$mtry), , drop = FALSE]
configs$config_id <- paste0("nt", configs$ntree, "_m", configs$mtry)

smoke <- "--smoke" %in% commandArgs(trailingOnly = TRUE)
cache_dir <- file.path(root, "data", "processed", "grrf_gamma1_grid_cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

if (smoke) {
  jobs <- data.frame(fold = 1L, config_id = configs$config_id[1])
  cat("【快速检查】只运行第1折、第1组参数；输出是检查用途，非最终结果。\n")
} else {
  jobs <- expand.grid(fold = 1:5, config_id = configs$config_id)
  cat("【正式网格】4种参数 × 5折。运行中断后可重复同一命令继续。\n")
}

for (i in seq_len(nrow(jobs))) {
  fold <- as.integer(jobs$fold[i]); cfg_id <- as.character(jobs$config_id[i])
  cfg <- configs[configs$config_id == cfg_id, , drop = FALSE]
  ntree <- as.integer(cfg$ntree[1]); mtry <- as.integer(cfg$mtry[1])
  cache_path <- file.path(cache_dir, sprintf("%s_fold%d.rds", cfg_id, fold))
  if (file.exists(cache_path)) {
    cat(sprintf("已缓存，跳过：%s / 折%d\n", cfg_id, fold))
    next
  }
  tr <- which(splits$validation_fold != fold)
  va <- which(splits$validation_fold == fold)
  stopifnot(length(intersect(unique(subjects[tr]), unique(subjects[va]))) == 0)
  Xtr <- X[tr, , drop = FALSE]
  ytr <- y[tr]
  seed <- as.integer(10000L + ntree + mtry * 10L + fold)

  set.seed(seed)
  start <- proc.time()[3]
  rf <- RRF(x = Xtr, y = ytr, ntree = ntree, mtry = mtry,
            flagReg = 0, importance = TRUE)
  rf_seconds <- as.numeric(proc.time()[3] - start)
  imp <- as.numeric(rf$importance[, "MeanDecreaseGini"])
  stopifnot(length(imp) == 561L, all(is.finite(imp)), max(imp) > 0)
  normalized_imp <- imp / max(imp)
  coef_reg <- (1 - gamma_grrf) + gamma_grrf * normalized_imp

  set.seed(seed + 1L)
  start <- proc.time()[3]
  grrf <- RRF(x = Xtr, y = ytr, ntree = ntree, mtry = mtry,
              flagReg = 1, coefReg = coef_reg)
  grrf_seconds <- as.numeric(proc.time()[3] - start)
  cols <- sort(unique(as.integer(grrf$feaSet))) - 1L  # 转Python的0基索引
  stopifnot(length(cols) > 0, all(cols >= 0 & cols < 561L))

  row <- list(fold = fold, config_id = cfg_id, ntree = ntree, mtry = mtry,
              gamma_grrf = gamma_grrf, seed = seed,
              feature_index_python = cols,
              n_features = length(cols),
              n_train_samples = length(tr), n_val_samples = length(va),
              rf_seconds = rf_seconds, grrf_seconds = grrf_seconds,
              selection_total_seconds = rf_seconds + grrf_seconds,
              r_version = as.character(getRversion()),
              rrf_version = as.character(packageVersion("RRF")))
  saveRDS(row, cache_path)
  cat(sprintf("完成 %-11s 折%d: %3d特征, RF %.2fs + GRRF %.2fs\n",
              cfg_id, fold, length(cols), rf_seconds, grrf_seconds))
}

if (smoke) {
  cat("【快速检查通过】现在可不带 --smoke 执行正式网格。\n")
  quit(save = "no")
}

feature_rows <- list(); time_rows <- list(); k <- 0L
for (j in seq_len(nrow(configs))) for (fold in 1:5) {
  cfg_id <- configs$config_id[j]
  p <- file.path(cache_dir, sprintf("%s_fold%d.rds", cfg_id, fold))
  if (!file.exists(p)) stop(paste("缺少中间结果：", p))
  r <- readRDS(p)
  stopifnot(identical(r$config_id, cfg_id), r$fold == fold)
  k <- k + 1L
  feature_rows[[k]] <- data.frame(
    fold = fold, config_id = cfg_id, ntree = r$ntree, mtry = r$mtry,
    gamma_grrf = r$gamma_grrf, feature_index_python = r$feature_index_python)
  time_rows[[k]] <- data.frame(
    fold = fold, config_id = cfg_id, ntree = r$ntree, mtry = r$mtry,
    gamma_grrf = r$gamma_grrf, seed = r$seed, n_features = r$n_features,
    rf_seconds = r$rf_seconds, grrf_seconds = r$grrf_seconds,
    selection_total_seconds = r$selection_total_seconds,
    n_train_samples = r$n_train_samples, n_val_samples = r$n_val_samples,
    r_version = r$r_version, rrf_version = r$rrf_version)
}
features <- do.call(rbind, feature_rows)
times <- do.call(rbind, time_rows)
write.csv(features, file.path(out_dir, "grrf_gamma1_grid_selected_features.csv"), row.names = FALSE)
write.csv(times, file.path(out_dir, "grrf_gamma1_grid_selection_times.csv"), row.names = FALSE)
cat("\n全部20个折-参数组合完成。已保存：\n")
cat("results/grrf_gamma1_grid_selected_features.csv\n")
cat("results/grrf_gamma1_grid_selection_times.csv\n")
cat("\n各组合选中特征平均值：\n")
print(aggregate(n_features ~ config_id, data = times, FUN = mean))
