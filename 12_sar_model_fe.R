# Imports
library(here)
source(here("utils/nested_cv", "sar.R"))

set.seed(42)

load_dot_env()
username <- Sys.getenv("DB_USERNAME")
password <- Sys.getenv("DB_PASSWORD")

db_connection_string <- sprintf("PG:dbname=greenspace host=localhost user=%s password=%s port=5432", username, password)

df <- st_read(db_connection_string, query = "SELECT * FROM split_engineered_dataset")

results <- evaluate_sar(df)

# Save output
write_json(results, "outputs/model_results/sar_fe.json", auto_unbox = TRUE, pretty = TRUE)
