options(warn = 1)
set.seed(42)

# Imports
library(spatialreg)
library(spdep)
library(sf)
library(Metrics)

# Load data
# complete_df <- st_read("datasets/3_combined/df.gpkg")
# train_df <- st_read("datasets/4_split/train_df.gpkg")
# test_df <- st_read("datasets/4_split/test_df.gpkg")
# train_df <- st_read("datasets/4_split/sample_train_df.gpkg")
# test_df <- st_read("datasets/4_split/sample_test_df.gpkg")
complete_df <- st_read("datasets/3_combined/df_with_fe.gpkg")
train_indices <- sample(nrow(complete_df), size = 0.8 * nrow(complete_df))
train_df <- complete_df[train_indices, ]
test_df <- complete_df[-train_indices, ]

# Extract fold ids
# train_fold_ids <- train_df$fold_id_r
# folds <- unique(train_fold_ids)

# Drop unneeded features
# train_drop_cols <- c(
#   "lsoa", "good_health", "fair_health",
#   "bad_health", "very_bad_health", "total_area",
#   "greenspace_area", "fold_id_python", "fold_id_r", "geometry"
# )
# test_drop_cols <- c(
#   "lsoa", "good_health", "fair_health",
#   "bad_health", "very_bad_health", "total_area",
#   "greenspace_area", "geometry"
# )

drop_cols <- c(
  "lsoa", "good_health", "fair_health",
  "bad_health", "very_bad_health", "total_area",
  "greenspace_area", "geometry", "prevalent_white_other"
)

train_df <- train_df[, !names(train_df) %in% drop_cols]
train_df <- st_drop_geometry(train_df)
test_df <- test_df[, !names(test_df) %in% drop_cols]
test_df <- st_drop_geometry(test_df)

# Create spatial weight matrix
create_spatial_weights <- function(df) {

  coords <- cbind(df$x_coord, df$y_coord) # Create coordinate feature
  knn <- knearneigh(coords, k = 4)  # Calculate k nearest neighbours using coords
  w <- nb2listw(knn2nb(knn), zero.policy = TRUE)  # Calculate weights

  # Return weights
  w

}

# Define build model function
build_model <- function(df) {

  # Create spatial weight matrix for train_df
  print("Creating spatial weight matrix for training df...")
  w <- create_spatial_weights(df)

  # Fit SAR model using spatial weights
  print("Fitting model...")
  system.time(
    sar_model <- lagsarlm(
      very_good_health ~ .,   # Predict very_good_health as a function of all other features
      data = df,
      listw = w,   # Spatial weights to calculate lagged dependent variable
      quiet = FALSE,
    )
  )

  # Return SAR model
  sar_model
  
}

# for (fold in folds) {

#   message <- paste("--- Training on fold ", fold, "---")
#   print(paste(message))

#   # Separate data into training and validation sets
#   is_in_validation_set <- train_fold_ids == fold
#   is_in_training_set <- !is_in_validation_set
#   fold_train_df <- train_df[is_in_training_set, ]
#   fold_validation_df <- train_df[is_in_validation_set, ]
#   rownames(fold_validation_df) <- NULL

#   # Build model
#   model <- build_model(fold_train_df)
  
#   # Create spatial weight matrix for complete_df
#   print("Creating spatial weight matrix for complete df...")
#   fold_validation_weights <- create_spatial_weights(fold_validation_df)

#   # Calculate predictions
#   print("Calculating predictions...")
#   predictions <- predict(
#     model,
#     newdata = fold_validation_df,
#     listw = fold_validation_weights,    # Spatial weights to calculate lagged dependent variable
#     zero.policy = TRUE
#   )  

#   # Calculate metrics
#   labels <- fold_validation_df$very_good_health
#   mae <- mae(labels, predictions)
#   mse <- mse(labels, predictions)
#   sum_of_squares_residual <- sum((labels - predictions) ^ 2)
#   sum_of_squares_total <- sum((labels - mean(labels)) ^ 2)
#   r2 <- 1 - (sum_of_squares_residual / sum_of_squares_total)

#   # Print metrics
#   results_message <- paste("--- Results for fold ", fold, "---")
#   print(results_message)
#   print(paste("MAE:", round(mae, 4)))
#   print(paste("MSE:", round(mse, 4)))
#   print(paste("R²:", round(r2, 4)))
#   print("")

#   # Remove large variables to avoid memory issues
#   rm(model, fold_validation_weights, predictions)
#   gc()

# }

# Build model
model <- build_model(train_df)

# Create spatial weight matrix for complete_df
print("Creating spatial weight matrix for test df...")
test_weights <- create_spatial_weights(test_df)

# Calculate predictions
print("Calculating predictions...")
predictions <- predict(
  model,
  newdata = test_df,
  listw = test_weights,    # Spatial weights to calculate lagged dependent variable
  zero.policy = TRUE
)  

# Calculate metrics
labels <- test_df$very_good_health
mae <- mae(labels, predictions)
mse <- mse(labels, predictions)
sum_of_squares_residual <- sum((labels - predictions) ^ 2)
sum_of_squares_total <- sum((labels - mean(labels)) ^ 2)
r2 <- 1 - (sum_of_squares_residual / sum_of_squares_total)

# Print metrics
print(results_message)
print(paste("MAE:", round(mae, 4)))
print(paste("MSE:", round(mse, 4)))
print(paste("R²:", round(r2, 4)))
print("")
