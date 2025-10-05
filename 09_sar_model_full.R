options(warn = 1)
set.seed(42)

# Imports
library(spatialreg)
library(spdep)
library(sf)
library(sfdep)
library(Metrics)

# Load data
df <- st_read("datasets/5_split/df_full.gpkg")
drop_cols <- names(df)[grepl("python", names(df))]
df <- df[, !names(df) %in% drop_cols]

# Extract fold ids
outer_fold_ids <- df$outer_loop_fold_id_r
outer_splits <- sort(unique(outer_fold_ids))
inner_fold_ids <- df[, grepl("inner_loop", names(df)), drop = FALSE]
inner_splits <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# Drop unneeded cols
drop_cols <- names(df)[grepl("fold_id", names(df))]
df <- df[, !names(df) %in% drop_cols]

# Function - Create spatial weight matrix
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

# Nested cross-validation loop
outer_cv_results <- vector(mode = "list", length = 0)
print(paste("outer_cv_results initialized:", exists("outer_cv_results")))

for (current_outer_split in outer_splits) {

  hp_combinations <- vector(mode = "list", length = 0)
  inner_cv_results <- vector(mode = "list", length = 0)

  # Separate data into training and validation sets
  is_in_validation_set <- outer_fold_ids == current_outer_split
  is_in_training_set <- !is_in_validation_set
  outer_train_df <- df[is_in_training_set, ]
  outer_validation_df <- df[is_in_validation_set, ]
  current_inner_fold_ids <- inner_fold_ids[is_in_training_set, , drop = FALSE]    # Removes rows not in current training set
  rownames(outer_train_df) <- NULL
  rownames(outer_validation_df) <- NULL   # Reset row numbers so they align with spatial weight matrix for predictions
  rownames(current_inner_fold_ids) <- NULL

  # Loop to test 10 hyperparameter combinations
  for (i in 1:10) {

    # Get random hyperparameters
    hps <- list()
    # weighting_method <- sample(c("distance", "knn", "queen", "rook"), 1)
    weighting_method <- "queen"
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
    hp_combinations <- c(hp_combinations, list(hps))

    # Inner cross-validation to evaluate hyperparameter combinations
    for (current_inner_split in inner_splits) {

      print(paste("--- Outer split ", current_outer_split, ": Training model ", i, " on inner split ", current_inner_split, "---"))

      # Separate data into training and validation sets
      is_in_validation_set <- current_inner_fold_ids[[paste0("inner_loop_", current_outer_split, "_fold_id_r")]] == current_inner_split
      is_in_training_set <- !is_in_validation_set
      inner_train_df <- outer_train_df[is_in_training_set, ]
      inner_validation_df <- outer_train_df[is_in_validation_set, ]
      rownames(inner_train_df) <- NULL
      rownames(inner_validation_df) <- NULL   # Reset row numbers so they align with spatial weight matrix for predictions

      # Build model using inner_train_df
      print("Building model...")
      model <- build_model(inner_train_df, hps)
      
      # Create spatial weight matrix for inner_validation_df
      print("Creating spatial weight matrix...")
      inner_validation_w <- create_spatial_weights(inner_validation_df, hps)

      # Calculate predictions on inner_validation_df
      print("Calculating predictions...")
      inner_validation_df_no_geom <- st_drop_geometry(inner_validation_df)

      predictions <- predict(
        model,
        newdata = inner_validation_df_no_geom,
        listw = inner_validation_w,
        zero.policy = TRUE
      )  

      # Calculate accuracy metrics
      print("Evaluating predictions...")
      labels <- inner_validation_df$very_good_health
      metrics <- get_evaluation_metrics(labels, predictions)
      mae <- metrics$mae
      mse <- metrics$mse
      r2 <- metrics$r2

      current_split_results <- list(
        hp_combination = i,
        inner_split = current_inner_split,
        hps = hps,
        mae = mae,
        mse = mse,
        r2 = r2
      )

      inner_cv_results <- c(inner_cv_results, list(current_split_results))

      # Remove large variables to avoid memory issues
      rm(model, inner_validation_w, predictions)
      gc()

    }

  }

  print(paste("--- Outer split ", current_outer_split, ": Training on optimised model"))

  optimal_hps <- get_optimal_hps(hp_combinations, inner_cv_results)

  # Build model using train_df
  print("Building model...")
  model <- build_model(outer_train_df, optimal_hps)

  # Create spatial weight matrix for validation_df
  print("Creating spatial weight matrix...")
  outer_validation_w <- create_spatial_weights(outer_validation_df, optimal_hps)

  # Calculate predictions on validation_df
  print("Calculating predictions...")
  outer_validation_df_no_geom <- st_drop_geometry(outer_validation_df)
  predictions <- predict(
    model,
    newdata = outer_validation_df_no_geom,
    listw = outer_validation_w,
    zero.policy = TRUE
  )  

  # Calculate accuracy metrics
  print("Evaluating predictions...")
  labels <- outer_validation_df$very_good_health
  metrics <- get_evaluation_metrics(labels, predictions)
  mae <- metrics$mae
  mse <- metrics$mse
  r2 <- metrics$r2

  current_split_results <- list(
    outer_split = current_outer_split,
    hps = optimal_hps,
    mae = mae,
    mse = mse,
    r2 = r2,
    inner_cv_results = inner_cv_results
  )

  outer_cv_results <- c(outer_cv_results, list(current_split_results))

  # Remove large variables to avoid memory issues
  rm(model, outer_validation_w, predictions)
  gc()

}