library(spatialreg)
library(spdep)
library(sf)

raw_complete_df <- read.csv("datasets/split/complete_df.csv")
raw_train_df <- read.csv("datasets/split/train_df.csv")
raw_test_df <- read.csv("datasets/split/test_df.csv")

coords <- cbind(raw_train_df$x_coord, raw_train_df$y_coord)
knn <- knearneigh(coords, k=4)
w <- nb2listw(knn2nb(knn))

sar_model <- lagsarlm(
    very_good_health ~ greenspace_proportion + imd,
    data = raw_train_df,
    listw = w
)

summary(sar_model)

complete_df_coords <- cbind(raw_complete_df$x_coord, raw_complete_df$y_coord)
complete_df_knn <- knearneigh(complete_df_coords, k=4)
complete_df_w <- nb2listw(knn2nb(complete_df_knn))

predictions <- predict(sar_model, raw_test_df, listw = complete_df_w)

raw_test_df$sar_predictions <- predictions

write.csv(raw_test_df, "datasets/results/test_df_sar.csv", row.names = FALSE)
