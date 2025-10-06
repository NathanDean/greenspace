# Function - Create spatial weight matrix
library(spdep)
library(sf)
library(sfdep)
library(Metrics)

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

  prepared_data

}

split_data <- function(df, current_split, fold_ids, is_outer = FALSE, inner_fold_ids = NULL) {
  is_in_validation_set <- fold_ids == current_split
  is_in_training_set <- !is_in_validation_set
  train_df <- df[is_in_training_set, ]
  val_df <- df[is_in_validation_set, ]
  current_inner_fold_ids <- NULL
  if (is_outer) {
    current_inner_fold_ids <- inner_fold_ids[is_in_training_set, , drop = FALSE]    # Removes rows not in current training set
    rownames(current_inner_fold_ids) <- NULL
  }
  rownames(train_df) <- NULL
  rownames(val_df) <- NULL   # Reset row numbers so they align with spatial weight matrix for predictions
  split_data <- list(
    train_df = train_df,
    val_df = val_df,
    current_inner_fold_ids = current_inner_fold_ids
  )

  split_data
}

get_random_hyperparameters <- function() {
  hps <- list()
  weighting_method <- sample(c("distance", "knn", "queen", "rook"), 1)
  hps$weighting_method <- weighting_method
  if (weighting_method == "distance") {
    max_distance <- sample(1000:2000, 1)
    hps$max_distance <- max_distance

  } else if (weighting_method == "knn") {
    k <- sample(2:30, 1)
    hps$k <- k
  } else {
    is_queen <- sample(c(TRUE, FALSE), 1)
    hps$is_queen <- is_queen
  }

  hps
  
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

# Function - Build model
build_model <- function(df, hps) {

  # Create spatial weight matrix for train_df
  print("Creating spatial weight matrix for train_df...")
  w <- create_spatial_weights(df, hps)

  df_no_geom <- st_drop_geometry(df)
  rownames(df_no_geom) <- NULL

  # Fit SAR model using spatial weights
  print("Fitting model...")
  system.time(
    sar_model <- lagsarlm(
      very_good_health ~ .,   # Predict very_good_health as a function of all other features apart from coords
      data = df_no_geom,
      listw = w,    # Spatial weights to calculate lagged dependent variable
      method = "LU",    # Use LU decomposition to calculate lagged dependent variable
      quiet = TRUE,
      zero.policy = TRUE
    )
  )

  # Return SAR model
  sar_model
  
}

# Function - Get evaluation metrics for set of predictions
get_evaluation_metrics <- function(labels, predictions) {

  mae <- mae(labels, predictions)
  mse <- mse(labels, predictions)
  sum_of_squares_residual <- sum((labels - predictions) ^ 2)
  sum_of_squares_total <- sum((labels - mean(labels)) ^ 2)
  r2 <- 1 - (sum_of_squares_residual / sum_of_squares_total)
  list(mae = mae, mse = mse, r2 = r2)

}

# Function - Average prediction scores for set of cross-validation results
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

# Function - Find hyperparameter combination that produced best results from a set of cross-validation results
get_optimal_hps <- function(hp_combinations, cv_results) {
  hp_combination_scores <- c()

  for (hp_combination in seq_along(hp_combinations)) {
    current_hp_combination_results <- cv_results[sapply(cv_results, function(result) result$hp_combination == hp_combination)]
    avg_scores <- get_avg_scores(current_hp_combination_results)
    mae <- avg_scores$mae
    mse <- avg_scores$mse
    r2 <- avg_scores$r2
    hp_combination_scores <- c(hp_combination_scores, mse)
  }

  optimal_combination <- which.min(hp_combination_scores)
  optimal_hps <- hp_combinations[[optimal_combination]]
  optimal_hps

}