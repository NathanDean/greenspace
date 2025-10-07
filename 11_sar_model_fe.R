# Set options and random seed
# options(warn = 1)
set.seed(42)

# Imports
library(spatialreg)
library(spdep)
library(sf)
library(sfdep)
library(Metrics)
library(jsonlite)
source("utils/sar_utils.R")

# Prepare data

## Load data
df <- st_read("datasets/5_split/df_fe.gpkg")

## Clean data and separate fold ids
prepared_data <- prepare_data(df)
outer_fold_ids <- prepared_data$outer_fold_ids
outer_splits <- prepared_data$outer_splits
inner_fold_ids <- prepared_data$inner_fold_ids
inner_splits <- prepared_data$inner_splits
df <- prepared_data$df

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

# Evaluate model

## Initialise results list
outer_cv_results <- vector(mode = "list", length = 0)

## Evaluate using nested cross-validation loop
print(paste("outer_cv_results initialized:", exists("outer_cv_results")))

for (current_outer_split in outer_splits) {

  hp_combinations <- vector(mode = "list", length = 0)
  inner_cv_results <- vector(mode = "list", length = 0)

  # Separate data into training and validation sets
  outer_split_data <- split_data(df, current_outer_split, outer_fold_ids, is_outer = TRUE, inner_fold_ids = inner_fold_ids)
  outer_train_df <- outer_split_data$train_df
  outer_val_df <- outer_split_data$val_df
  current_inner_fold_ids <- outer_split_data$current_inner_fold_ids

  # Loop to test 10 hyperparameter combinations
  for (i in 1:8) {

    # Get random hyperparameters
    hps <- get_random_hyperparameters()
    hp_combinations <- c(hp_combinations, list(hps))

    # Inner cross-validation to evaluate hyperparameter combinations
    for (current_inner_split in inner_splits) {

      print(paste("--- Outer split ", current_outer_split, ": Training model ", i, " on inner split ", current_inner_split, "---"))

      # Separate data into training and validation sets
      inner_split_data <- split_data(df, current_inner_split, current_inner_fold_ids[[paste0("inner_loop_", current_outer_split, "_fold_id_r")]])
      inner_train_df <- inner_split_data$train_df
      inner_val_df <- inner_split_data$val_df

      # Build model using inner_train_df
      print("Building model...")
      model <- build_model(inner_train_df, hps)
      
      # Create spatial weight matrix for inner_valn_df
      print("Creating spatial weight matrix...")
      inner_val_w <- create_spatial_weights(inner_val_df, hps)

      # Calculate predictions on inner_val_df
      print("Calculating predictions...")
      inner_val_df_no_geom <- st_drop_geometry(inner_val_df)

      predictions <- predict(
        model,
        newdata = inner_val_df_no_geom,
        listw = inner_val_w,
        zero.policy = TRUE
      )  

      # Calculate accuracy metrics
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

  optimal_hps <- get_optimal_hps(hp_combinations, inner_cv_results)

  # Build model using train_df
  print("Building model...")
  model <- build_model(outer_train_df, optimal_hps)

  # Create spatial weight matrix for validation_df
  print("Creating spatial weight matrix...")
  outer_val_w <- create_spatial_weights(outer_val_df, optimal_hps)

  # Calculate predictions on validation_df
  print("Calculating predictions...")
  outer_val_df_no_geom <- st_drop_geometry(outer_val_df)
  predictions <- predict(
    model,
    newdata = outer_val_df_no_geom,
    listw = outer_val_w,
    zero.policy = TRUE
  )  

  # Calculate accuracy metrics
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

# Save output
write_json(outer_cv_results, "outputs/model_results/sar_fe.json", auto_unbox = TRUE, pretty = TRUE)