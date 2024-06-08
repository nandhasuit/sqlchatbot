from langchain_community.utilities import SQLDatabase
import urllib


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    password_encoded = urllib.parse.quote_plus(password)
    db_uri = f"mysql+mysqlconnector://{user}:{password_encoded}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)