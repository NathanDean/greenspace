# Imports
library(here)
source(here("utils", "sar.R"))
source(here("utils/nested_cv", "sar.R"))

# Set random seed
set.seed(42)

df <- st_read(db_connection_string, query = "SELECT * FROM split_full_dataset")

results <- evaluate_sar(df)

# Save output
write_json(results, "outputs/model_results/sar_fe.json", auto_unbox = TRUE, pretty = TRUE)
