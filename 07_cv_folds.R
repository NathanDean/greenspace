# Imports
library(sf)
library(blockCV)
library(dotenv)
library(here)
source(here("utils", "db_utils.R"))

# Set random seed
set.seed(42)

# Load data and set geometries
df_full <- st_read(db_connection_string, query = "SELECT * FROM full_dataset")
st_geometry(df_full) <- "geometry"
df_fe <- st_read(db_connection_string, query = "SELECT * FROM engineered_dataset")
st_geometry(df_fe) <- "geometry"

# Measure distance of spatial autocorrelation for very_good_health
autocorrelation_info <- cv_spatial_autocor(
  x = df_full,
  column = "very_good_health"
)

# Set spatial block size > autocorrelation distance
autocorrelation_range <- autocorrelation_info$range_table$range
block_size <- autocorrelation_range * 2

# Create outer cross-validation folds using spatial blocking
outer_cv_folds <- cv_spatial(
  x = df_full,
  k = 5,
  size = block_size,
  iteration = 100
)

# Add outer fold allocations to dfs
df_full$outer_loop_fold_id_r <- outer_cv_folds$folds_ids # R indexes from 1
df_fe$outer_loop_fold_id_r <- outer_cv_folds$folds_ids
df_full$outer_loop_fold_id_python <- outer_cv_folds$folds_ids - 1 # Python indexes from 0
df_fe$outer_loop_fold_id_python <- outer_cv_folds$folds_ids - 1

# Create inner cross-validation folds
for (fold in unique(outer_cv_folds$folds_ids)) {
  # Get training set for current fold
  is_in_validation_set <- outer_cv_folds$folds_ids == fold
  is_in_training_set <- !is_in_validation_set
  training_set <- df_full[is_in_training_set, ]

  inner_cv_folds <- cv_spatial(
    x = training_set,
    k = 5,
    size = block_size,
    iteration = 100
  )

  # Set inner fold ids to NA
  df_full[[paste0("inner_loop_", fold, "_fold_id_r")]] <- NA
  df_fe[[paste0("inner_loop_", fold, "_fold_id_r")]] <- NA
  df_full[[paste0("inner_loop_", fold, "_fold_id_python")]] <- NA
  df_fe[[paste0("inner_loop_", fold, "_fold_id_python")]] <- NA

  # Add inner fold ids to training rows
  df_full[is_in_training_set, paste0("inner_loop_", fold, "_fold_id_r")] <- inner_cv_folds$folds_ids # R indexes from 1
  df_fe[is_in_training_set, paste0("inner_loop_", fold, "_fold_id_r")] <- inner_cv_folds$folds_ids
  df_full[is_in_training_set, paste0("inner_loop_", fold, "_fold_id_python")] <- inner_cv_folds$folds_ids - 1 # Python indexes from 0
  df_fe[is_in_training_set, paste0("inner_loop_", fold, "_fold_id_python")] <- inner_cv_folds$folds_ids - 1
}

# Save output
st_write(df_full, db_connection_string, "split_full_dataset", layer_options = "GEOMETRY_NAME=geometry", delete_layer = TRUE)
st_write(df_fe, db_connection_string, "split_engineered_dataset", layer_options = "GEOMETRY_NAME=geometry", delete_layer = TRUE)
