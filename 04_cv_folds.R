library(blockCV)
library(sf)

raw_df <- st_read("datasets/combined/lsoa_greenspace.gpkg")
st_geometry(raw_df) <- "geometry"

autocorrelation_info <- cv_spatial_autocor(
    x = raw_df,
    column = "very_good_health"
)

autocorrelation_range <- autocorrelation_info$range_table$range
block_size = autocorrelation_range * 2

spatial_blocks <- cv_spatial(
    x = raw_df,
    k = 10,
    size = block_size,
    iteration = 100
)

folds <- spatial_blocks$folds_list

raw_df$fold_id_r <- spatial_blocks$folds_ids
raw_df$fold_id_python <- raw_df$fold_id_r - 1

st_write(raw_df, "datasets/spatial_folds/df_with_folds.csv", delete_dsn = TRUE)