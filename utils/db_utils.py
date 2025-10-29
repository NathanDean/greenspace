import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
engine = create_engine(f"postgresql://{username}:{password}@localhost:5432/greenspace")
