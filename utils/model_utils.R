# Function - Create spatial weight matrix
library(spdep)
library(sf)
library(sfdep)
library(Metrics)

# Separate cross-validation fold ids from df and drop unneeded columns
prepare_data <- function(df) {
  drop_cols <- names(df)[grepl("python", names(df))]
  df <- df[, !names(df) %in% drop_cols]

  # Extract fold ids
  outer_fold_ids <- df$outer_loop_fold_id_r
  outer_splits <- sort(unique(outer_fold_ids))
  inner_fold_ids <- df[, grepl("inner_loop", names(df)), drop = FALSE]
  inner_splits <- c(1, 2, 3, 4, 5)

  # Drop unneeded cols
  drop_cols <- names(df)[grepl("fold_id", names(df))]
  df <- df[, !names(df) %in% drop_cols]

  prepared_data <- list(
    outer_fold_ids = outer_fold_ids,
    outer_splits = outer_splits,
    inner_fold_ids = inner_fold_ids,
    inner_splits = inner_splits,
    df = df
  )

  # Return separated data
  prepared_data
}


# Split df into training and validation sets
split_data <- function(df, current_split, fold_ids, is_outer = FALSE, inner_fold_ids = NULL) {
  is_in_validation_set <- fold_ids == current_split
  is_in_training_set <- !is_in_validation_set
  train_df <- df[is_in_training_set, ]
  val_df <- df[is_in_validation_set, ]

  # If splitting data for outer cross-validation loop, get fold ids for inner cross-validation loop
  current_inner_fold_ids <- NULL
  if (is_outer) {
    current_inner_fold_ids <- inner_fold_ids[is_in_training_set, , drop = FALSE] # Removes rows not in current training set
    rownames(current_inner_fold_ids) <- NULL # Reset row numbers
  }
  rownames(train_df) <- NULL
  rownames(val_df) <- NULL

  split_data <- list(
    train_df = train_df,
    val_df = val_df,
    current_inner_fold_ids = current_inner_fold_ids
  )

  # Return split data
  split_data
}


create_spatial_weights <- function(df, hps) {
  weighting_method <- hps$weighting_method
  coords <- st_centroid(st_geometry(df))
  w <- NULL

  if (weighting_method == "distance") {
    print(weighting_method)
    print(hps$max_distance)
    dnn <- dnearneigh(coords, d1 = 200, d2 = hps$max_distance)
    w <- nb2listw(dnn, zero.policy = TRUE)
  } else if (weighting_method == "knn") {
    print(weighting_method)
    print(hps$k)
    knn <- knearneigh(coords, k = hps$k)
    w <- nb2listw(knn2nb(knn), zero.policy = TRUE)
  } else {
    print(weighting_method)
    print(hps$is_queen)
    nb <- st_contiguity(df, hps$is_queen)
    w <- nb2listw(nb, zero.policy = TRUE)
  }

  # Return weights
  w
}

get_evaluation_metrics <- function(labels, predictions) {
  mae <- mae(labels, predictions)
  mse <- mse(labels, predictions)
  sum_of_squares_residual <- sum((labels - predictions)^2)
  sum_of_squares_total <- sum((labels - mean(labels))^2)
  r2 <- 1 - (sum_of_squares_residual / sum_of_squares_total)
  list(mae = mae, mse = mse, r2 = r2)
}

# Get average accuracy metrics from set of cross-validation results
get_avg_scores <- function(cv_results) {
  mae_scores <- c()
  mse_scores <- c()
  r2_scores <- c()

  for (result in cv_results) {
    mae_scores <- c(mae_scores, result$mae)
    mse_scores <- c(mse_scores, result$mse)
    r2_scores <- c(r2_scores, result$r2)
  }

  mae <- mean(mae_scores)
  mse <- mean(mse_scores)
  r2 <- mean(r2_scores)

  avgs <- list(mae = mae, mse = mse, r2 = r2)
  avgs
}

# Get best performing hyperparameters from set of cross-validation results
get_optimal_hps <- function(hp_combinations, cv_results) {
  hp_combination_scores <- c()

  for (hp_combination in seq_along(hp_combinations)) {
    current_hp_combination_results <- cv_results[sapply(cv_results, function(result) result$hp_combination == hp_combination)]
    avg_scores <- get_avg_scores(current_hp_combination_results)
    mae <- avg_scores$mae
    mse <- avg_scores$mse
    r2 <- avg_scores$r2
    hp_combination_scores <- c(hp_combination_scores, mae)
  }

  optimal_combination <- which.min(hp_combination_scores)
  optimal_hps <- hp_combinations[[optimal_combination]]
  optimal_hps
}
