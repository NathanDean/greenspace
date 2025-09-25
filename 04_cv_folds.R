# Imports
library(blockCV)
library(sf)

# Set randomness seed
set.seed(42)

# Load data
complete_df <- st_read("datasets/3_combined/df.gpkg")
st_geometry(complete_df) <- "geometry"

# Split data into train/test sets
training_indices <- sample(
  nrow(complete_df),
  size = 0.8 * nrow(complete_df)
)
train_df <- complete_df[training_indices, ]
test_df <- complete_df[-training_indices, ]

# Measure distance of spatial autocorrelation for very_good_health
autocorrelation_info <- cv_spatial_autocor(
  x = train_df,
  column = "very_good_health"
)

# Set spatial block size > autocorrelation distance
autocorrelation_range <- autocorrelation_info$range_table$range
block_size <- autocorrelation_range * 2

# Create cross-validation folds using spatial blocking
cv_folds <- cv_spatial(
  x = train_df,
  k = 10,
  size = block_size,
  iteration = 100
)

# Add fold allocation to train_df rows
train_df$fold_id_r <- cv_folds$folds_ids            # R indexes from 1
train_df$fold_id_python <- train_df$fold_id_r - 1   # Python indexes from 0

# Save output
st_write(train_df, "datasets/4_split/train_df.gpkg", delete_dsn = TRUE)
st_write(test_df, "datasets/4_split/test_df.gpkg", delete_dsn = TRUE)