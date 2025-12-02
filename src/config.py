"""Configuration management for the PUMS data collector."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "pums_user")
    password: str = os.getenv("POSTGRES_PASSWORD", "pums_password")
    database: str = os.getenv("POSTGRES_DB", "pums_db")

    @property
    def connection_string(self) -> str:
        """Return SQLAlchemy connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class CensusConfig:
    """Census API configuration."""
    api_key: str = os.getenv("CENSUS_API_KEY", "")
    base_url: str = "https://api.census.gov/data"
    year: int = int(os.getenv("ACS_YEAR", "2022"))
    dataset: str = os.getenv("ACS_DATASET", "acs/acs1/pums")

    @property
    def api_url(self) -> str:
        """Return full API URL for the configured dataset."""
        return f"{self.base_url}/{self.year}/{self.dataset}"


# Global config instances
db_config = DatabaseConfig()
census_config = CensusConfig()
