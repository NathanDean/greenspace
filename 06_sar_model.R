# Imports
library(spatialreg)
library(spdep)
library(sf)

# Load data
complete_df <- st_read("datasets/3_combined/df.gpkg")
train_df <- st_read("datasets/4_split/train_df.gpkg")
test_df <- st_read("datasets/4_split/test_df.gpkg")

# Create spatial weight matrix for train_df
coords <- cbind(train_df$x_coord, train_df$y_coord) # Create coordinate feature
knn <- knearneigh(coords, k=4)  # Calculate k nearest neighbours using coords
w <- nb2listw(knn2nb(knn))  # Calculate weights

# Fit SAR model using spatial weights
sar_model <- lagsarlm(
  very_good_health ~ greenspace_proportion + imd,
  data = train_df,
  listw = w
)

# Create spatial weight matrix for complete_df
complete_df_coords <- cbind(complete_df$x_coord, complete_df$y_coord)
complete_df_knn <- knearneigh(complete_df_coords, k=4)
complete_df_w <- nb2listw(knn2nb(complete_df_knn))

# Calculate predictions
predictions <- predict(sar_model, test_df, listw = complete_df_w)

# Add predicted values to test_df
test_df$sar_predictions <- predictions

# Save output
write.csv(test_df, "datasets/results/test_df_sar.csv", row.names = FALSE)