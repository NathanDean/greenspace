# Imports
library(spatialreg)
library(spdep)
library(sf)
library(sfdep)
library(Metrics)
library(jsonlite)
library(here)
source(here("utils", "model_utils.R"))

build_model <- function(df, hps) {
  print("Creating spatial weight matrix for train_df...")
  w <- create_spatial_weights(df, hps)

  # Drop geometry column
  df_no_geom <- st_drop_geometry(df)
  rownames(df_no_geom) <- NULL

  print("Fitting model...")
  system.time(
    sar_model <- lagsarlm(
      very_good_health ~ ., # Predict very_good_health as a function of all other features
      data = df_no_geom,
      listw = w,
      method = "LU",
      quiet = TRUE,
      zero.policy = TRUE
    )
  )

  # Return SAR model
  sar_model
}

get_random_hyperparameters <- function() {
  hps <- list()

  # Select weighting method
  weighting_method <- sample(c("distance", "knn", "queen", "rook"), 1)
  hps$weighting_method <- weighting_method

  # Select hyperparameters for selected weighting method
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

  # Return hyperparameters
  hps
}

evaluate_sar <- function(df) {
  outer_cv_results <- vector(mode = "list", length = 0)

  # Clean data and separate fold ids
  prepared_data <- prepare_data(df)
  outer_fold_ids <- prepared_data$outer_fold_ids
  outer_splits <- prepared_data$outer_splits
  inner_fold_ids <- prepared_data$inner_fold_ids
  inner_splits <- prepared_data$inner_splits
  df <- prepared_data$df

  # Outer cross-validation loop to evaluate model
  for (current_outer_split in outer_splits) {
    hp_combinations <- vector(mode = "list", length = 0)
    inner_cv_results <- vector(mode = "list", length = 0)

    # Separate data into training and validation sets
    outer_split_data <- split_data(df, current_outer_split, outer_fold_ids, is_outer = TRUE, inner_fold_ids = inner_fold_ids)
    outer_train_df <- outer_split_data$train_df
    outer_val_df <- outer_split_data$val_df
    current_inner_fold_ids <- outer_split_data$current_inner_fold_ids

    # Loop to test 8 hyperparameter combinations
    for (i in 1:8) {
      # Get random hyperparameters
      hps <- get_random_hyperparameters()
      hp_combinations <- c(hp_combinations, list(hps))

      # Inner cross-validation to select model
      for (current_inner_split in inner_splits) {
        print(paste("--- Outer split ", current_outer_split, ": Training model ", i, " on inner split ", current_inner_split, "---"))

        # Separate data into training and validation sets
        inner_split_data <- split_data(df, current_inner_split, current_inner_fold_ids[[paste0("inner_loop_", current_outer_split, "_fold_id_r")]])
        inner_train_df <- inner_split_data$train_df
        inner_val_df <- inner_split_data$val_df

        print("Building model...")
        model <- build_model(inner_train_df, hps)

        print("Creating spatial weight matrix...")
        inner_val_w <- create_spatial_weights(inner_val_df, hps)

        print("Calculating predictions...")
        inner_val_df_no_geom <- st_drop_geometry(inner_val_df)

        predictions <- predict(
          model,
          newdata = inner_val_df_no_geom,
          listw = inner_val_w,
          zero.policy = TRUE
        )

        print("Evaluating predictions...")
        labels <- inner_val_df$very_good_health
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
        rm(model, inner_val_w, predictions)
        gc()
      }
    }

    print(paste("--- Outer split ", current_outer_split, ": Training on optimised model"))

    # Get optimal hps for current training set
    optimal_hps <- get_optimal_hps(hp_combinations, inner_cv_results)

    print("Building model...")
    model <- build_model(outer_train_df, optimal_hps)

    print("Creating spatial weight matrix...")
    outer_val_w <- create_spatial_weights(outer_val_df, optimal_hps)

    print("Calculating predictions...")
    outer_val_df_no_geom <- st_drop_geometry(outer_val_df)
    predictions <- predict(
      model,
      newdata = outer_val_df_no_geom,
      listw = outer_val_w,
      zero.policy = TRUE
    )

    print("Evaluating predictions...")
    labels <- outer_val_df$very_good_health
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
    rm(model, outer_val_w, predictions)
    gc()
  }

  # Return results
  outer_cv_results
}
