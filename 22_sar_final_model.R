library(spatialreg)
library(spdep)
library(sf)
library(sfdep)
library(jsonlite)
library(dotenv)
library(here)
source(here("utils", "db_utils.R"))
source(here("utils/nested_cv", "sar.R"))

set.seed(42)

# Prepare data
df <- st_read(db_connection_string, query = "SELECT * FROM engineered_dataset")
coords <- st_centroid(st_geometry(df))

# Create spatial weight matrix
print("Creating spatial weight matrix...")
model_knn <- knearneigh(coords, k = 5)
model_w <- nb2listw(knn2nb(model_knn), zero.policy = TRUE)

# Drop geometry column
df_no_geom <- st_drop_geometry(df)
rownames(df_no_geom) <- NULL

print("Fitting model...")
system.time(
  model <- lagsarlm(
    very_good_health ~ ., # Predict very_good_health as a function of all other features
    data = df_no_geom,
    listw = model_w,
    method = "LU",
    quiet = TRUE,
    zero.policy = TRUE
  )
)

print("Getting predictions...")
predictions <- predict(
  model,
  newdata = NULL,
  listw = model_w,
  zero.policy = TRUE
)

print("Calculating residuals...")
residuals <- df$very_good_health - predictions

print("Calculating Moran's I")
moran_knn <- knearneigh(coords, k = 8)
moran_w <- nb2listw(knn2nb(moran_knn), zero.policy = TRUE)
moran_results <- moran.test(x = residuals, listw = moran_w)

print(moran_results)
