# Select features on the complete official training set; never read test data.
.libPaths(c(file.path(getwd(), ".r-library"), .libPaths()))
suppressPackageStartupMessages(library(RRF))
root <- normalizePath(getwd(), winslash = "/")
train <- file.path(root, "data", "raw", "UCI HAR Dataset", "train")
out <- file.path(root, "results", "official_test_grrf")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
X <- as.matrix(read.table(file.path(train, "X_train.txt"), header = FALSE))
y <- factor(scan(file.path(train, "y_train.txt"), quiet = TRUE))
stopifnot(nrow(X) == 7352L, ncol(X) == 561L, length(y) == 7352L)

# Fixed before test evaluation, from the existing training-only five-fold work.
configs <- data.frame(
  method = c("GRRF_gamma025_nt100_m23", "GRRF_gamma1_nt500_m23"),
  gamma_grrf = c(0.25, 1.0), ntree = c(100L, 500L), mtry = c(23L, 23L),
  seed = c(20261025L, 20261525L))
feature_rows <- list()
time_rows <- list()
for (i in seq_len(nrow(configs))) {
  cfg <- configs[i, ]
  set.seed(cfg$seed)
  start <- proc.time()[3]
  rf <- RRF(x = X, y = y, ntree = cfg$ntree, mtry = cfg$mtry,
            flagReg = 0, importance = TRUE)
  rf_seconds <- as.numeric(proc.time()[3] - start)
  imp <- as.numeric(rf$importance[, "MeanDecreaseGini"])
  stopifnot(length(imp) == 561L, all(is.finite(imp)), max(imp) > 0)
  coef_reg <- (1 - cfg$gamma_grrf) + cfg$gamma_grrf * imp / max(imp)
  set.seed(cfg$seed + 1L)
  start <- proc.time()[3]
  grrf <- RRF(x = X, y = y, ntree = cfg$ntree, mtry = cfg$mtry,
              flagReg = 1, coefReg = coef_reg)
  grrf_seconds <- as.numeric(proc.time()[3] - start)
  selected <- sort(unique(as.integer(grrf$feaSet))) - 1L
  stopifnot(length(selected) > 0, all(selected >= 0L & selected < 561L))
  feature_rows[[i]] <- data.frame(method = cfg$method, feature_index_python = selected)
  time_rows[[i]] <- data.frame(method = cfg$method, gamma_grrf = cfg$gamma_grrf,
                               ntree = cfg$ntree, mtry = cfg$mtry, seed = cfg$seed,
                               n_features = length(selected), rf_seconds = rf_seconds,
                               grrf_seconds = grrf_seconds,
                               selection_seconds = rf_seconds + grrf_seconds,
                               r_version = as.character(getRversion()),
                               rrf_version = as.character(packageVersion("RRF")))
  cat(sprintf("%s: %d features, selection %.1f s\n", cfg$method,
              length(selected), rf_seconds + grrf_seconds), file = stderr())
}
write.csv(do.call(rbind, feature_rows), file.path(out, "selected_features.csv"), row.names = FALSE)
write.csv(do.call(rbind, time_rows), file.path(out, "selection_times.csv"), row.names = FALSE)
