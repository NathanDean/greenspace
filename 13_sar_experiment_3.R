# Imports
library(here)
source(here("utils", "sar.R"))
source(here("utils/nested_cv", "sar.R"))

set.seed(42)

df <- st_read(db_connection_string, query = "SELECT * FROM split_engineered_dataset")
df <- df[, !grepl("prevalent", names(df))]

results <- evaluate_sar(df)

# Save output
write_json(outer_cv_results, "outputs/model_results/sar_fe_reduced.json", auto_unbox = TRUE, pretty = TRUE)
