library(dotenv)

load_dot_env()
username <- Sys.getenv("DB_USERNAME")
password <- Sys.getenv("DB_PASSWORD")

db_connection_string <- sprintf("PG:dbname=greenspace host=localhost user=%s password=%s port=5432", username, password)
