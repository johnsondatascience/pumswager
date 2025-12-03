"""Configuration management for the PUMS data collector."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CensusConfig:
    """Census API configuration."""
    api_key: str = os.getenv("CENSUS_API_KEY", "")
    base_url: str = "https://api.census.gov/data"
    year: int = int(os.getenv("ACS_YEAR", "2023"))
    dataset: str = os.getenv("ACS_DATASET", "acs/acs1/pums")

    @property
    def api_url(self) -> str:
        """Return full API URL for the configured dataset."""
        return f"{self.base_url}/{self.year}/{self.dataset}"


# Global config instance
census_config = CensusConfig()
