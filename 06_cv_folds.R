# Imports
library(blockCV)
library(sf)

# Set random seed
set.seed(42)

# Load data and convert to spatial dataframes
df_full <- read.csv("datasets/4_fe/df_full.csv")
df_full$x <- df_full$x_coord
df_full$y <- df_full$y_coord
df_full <- st_as_sf(df_full, coords = c("x", "y"))
st_crs(df_full) <- "EPSG:3035"

df_fe <- read.csv("datasets/4_fe/df_fe.csv")
df_fe$x <- df_fe$x_coord
df_fe$y <- df_fe$y_coord
df_fe <- st_as_sf(df_fe, coords = c("x", "y"))
st_crs(df_fe) <- "EPSG:3035"

df_low_vif <- read.csv("datasets/4_fe/df_low_vif.csv")
df_low_vif$x <- df_low_vif$x_coord
df_low_vif$y <- df_low_vif$y_coord
df_low_vif <- st_as_sf(df_low_vif, coords = c("x", "y"))
st_crs(df_low_vif) <- "EPSG:3035"

# Measure distance of spatial autocorrelation for very_good_health
autocorrelation_info <- cv_spatial_autocor(
  x = df_full,
  column = "very_good_health"
)

# Set spatial block size > autocorrelation distance
autocorrelation_range <- autocorrelation_info$range_table$range
block_size <- autocorrelation_range * 2

# Create cross-validation folds using spatial blocking
cv_folds <- cv_spatial(
  x = df_full,
  k = 10,
  size = block_size,
  iteration = 100
)

# Add fold allocations to dfs
df_full$fold_id_r <- cv_folds$folds_ids            # R indexes from 1
df_fe$fold_id_r <- cv_folds$folds_ids
df_low_vif$fold_id_r <- cv_folds$folds_ids
df_full$fold_id_python <- df_full$fold_id_r - 1   # Python indexes from 0
df_fe$fold_id_python <- df_fe$fold_id_r - 1
df_low_vif$fold_id_python <- df_low_vif$fold_id_r - 1

df_full <- st_drop_geometry(df_full)
df_fe <- st_drop_geometry(df_fe)
df_low_vif <- st_drop_geometry(df_low_vif)

# Save output
write.csv(df_full, "datasets/5_split/df_full.csv", row.names = FALSE)
write.csv(df_fe, "datasets/5_split/df_fe.csv", row.names = FALSE)
write.csv(df_low_vif, "datasets/5_split/df_low_vif.csv", row.names = FALSE)