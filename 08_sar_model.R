options(warn = 1)
set.seed(42)

# Imports
library(spatialreg)
library(spdep)
library(sf)
library(Metrics)

# Load data
df <- read.csv("datasets/5_split/df_fe.csv")

# Extract fold ids
fold_ids <- df$fold_id_r
folds <- unique(fold_ids)

# Drop unneeded cols
drop_cols <- c("fold_id_python", "fold_id_r")
df <- df[, !names(df) %in% drop_cols]

# Function to create spatial weight matrix
create_spatial_weights <- function(df) {

  coords <- cbind(df$x_coord, df$y_coord)   # Create coordinate feature
  knn <- knearneigh(coords, k = 4)    # Calculate k nearest neighbours using coords
  w <- nb2listw(knn2nb(knn), zero.policy = TRUE)    # Calculate weights

  # Return weights
  w

}

# Function to build model
build_model <- function(df) {

  # Create spatial weight matrix for train_df
  print("Creating spatial weight matrix for training df...")
  w <- create_spatial_weights(df)

  # Fit SAR model using spatial weights
  print("Fitting model...")
  system.time(
    sar_model <- lagsarlm(
      very_good_health ~ . - x_coord - y_coord,   # Predict very_good_health as a function of all other features
      data = df,
      listw = w,    # Spatial weights to calculate lagged dependent variable
      method = "LU",    # Use LU decomposition to calculate lagged dependent variable
      quiet = FALSE,
    )
  )

  # Return SAR model
  sar_model
  
}

for (fold in folds) {

  message <- paste("--- Training on fold ", fold, "---")
  print(paste(message))

  # Separate data into training and validation sets
  is_in_validation_set <- fold_ids == fold
  is_in_training_set <- !is_in_validation_set
  train_df <- df[is_in_training_set, ]
  validation_df <- df[is_in_validation_set, ]
  rownames(validation_df) <- NULL   # Reset row numbers so they align with spatial weight matrix for predictions

  # Build model on train_df
  model <- build_model(train_df)
  
  # Create spatial weight matrix for validation_df
  print("Creating spatial weight matrix for validation df...")
  validation_w <- create_spatial_weights(validation_df)

  # Calculate predictions on validation_df
  print("Calculating predictions...")
  predictions <- predict(
    model,
    newdata = validation_df,
    listw = validation_w,
    zero.policy = TRUE
  )  

  # Calculate accuracy metrics
  labels <- validation_df$very_good_health
  mae <- mae(labels, predictions)
  mse <- mse(labels, predictions)
  sum_of_squares_residual <- sum((labels - predictions) ^ 2)
  sum_of_squares_total <- sum((labels - mean(labels)) ^ 2)
  r2 <- 1 - (sum_of_squares_residual / sum_of_squares_total)

  # Print metrics
  results_message <- paste("--- Results for fold ", fold, "---")
  print(results_message)
  print(paste("MAE:", round(mae, 4)))
  print(paste("MSE:", round(mse, 4)))
  print(paste("R²:", round(r2, 4)))
  print("")

  # Remove large variables to avoid memory issues
  rm(model, validation_w, predictions)
  gc()

}