"""Census API client for fetching ACS PUMS data."""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import census_config

logger = logging.getLogger(__name__)


# FIPS state codes for all US states and territories
STATE_CODES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming", "72": "Puerto Rico",
}

# Person-level variables to collect
PERSON_VARIABLES = [
    "SERIALNO",  # Serial number (links to household)
    "SPORDER",   # Person number within household
    "PUMA",      # Public Use Microdata Area (geographic)
    "STATE",     # State code (renamed from ST in 2023)
    
    # Demographics
    "AGEP",      # Age (proxy for experience: experience ≈ age - education_years - 6)
    "SEX",       # Sex (1=Male, 2=Female)
    "RAC1P",     # Race
    "HISP",      # Hispanic origin
    
    # Education
    "SCHL",      # Educational attainment (detailed: 1-24 scale)
    "FOD1P",     # Field of degree - first entry
    "FOD2P",     # Field of degree - second entry
    "SCIENGP",   # Science/engineering degree flag
    "SCIENGRLP", # Science/engineering related degree flag
    
    # Employment & Occupation
    "ESR",       # Employment status recode
    "COW",       # Class of worker
    "OCCP",      # Occupation code (SOC-based)
    "SOCP",      # SOC occupation code (detailed)
    "NAICSP",    # NAICS industry code (detailed)
    "INDP",      # Industry code
    "WKHP",      # Usual hours worked per week
    "WKWN",      # Weeks worked past 12 months (renamed from WKW in 2019+)
    "WKL",       # When last worked
    "JWMNP",     # Travel time to work (minutes)
    "JWTRNS",    # Means of transportation to work
    
    # Income & Earnings
    "WAGP",      # Wages/salary income past 12 months
    "SEMP",      # Self-employment income
    "PINCP",     # Total person income
    "PERNP",     # Total person earnings
    "OIP",       # Other income
    "INTP",      # Interest/dividends/rental income
    
    # Weight
    "PWGTP",     # Person weight (for population estimates)
]

# Household-level variables to collect (2023 ACS PUMS variable names)
HOUSEHOLD_VARIABLES = [
    "SERIALNO",  # Serial number
    "PUMA",      # Public Use Microdata Area
    "STATE",     # State code (renamed from ST in 2023)
    "TYPEHUGQ",  # Type of unit (renamed from TYPE in 2023)
    "BLD",       # Building type
    "TEN",       # Tenure
    "NP",        # Number of persons
    "HHT",       # Household type
    "HHT2",      # Household type (detailed)
    "HUPAC",     # HH presence and age of children (renamed from HUPAOC)
    "HINCP",     # Household income
    "FINCP",     # Family income
    "GRNTP",     # Gross rent
    "SMOCP",     # Owner costs
    "GRPIP",     # Rent as % income
    "OCPIP",     # Owner costs as % income
    "WGTP",      # Housing weight
]


class CensusAPIClient:
    """Client for interacting with the Census Bureau API."""

    def __init__(self, api_key: Optional[str] = None, year: Optional[int] = None):
        """Initialize the Census API client.
        
        Args:
            api_key: Census API key. If not provided, uses config.
            year: Survey year. If not provided, uses config.
        """
        self.api_key = api_key or census_config.api_key
        self.year = year or census_config.year
        self.base_url = census_config.base_url
        
        # Set up session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _build_url(self, record_type: str = "person") -> str:
        """Build the API URL for the specified record type.
        
        Args:
            record_type: Either 'person' or 'household'.
            
        Returns:
            Full API URL.
        """
        # ACS 1-year PUMS uses different endpoints for person vs housing
        if record_type == "person":
            return f"{self.base_url}/{self.year}/acs/acs1/pums"
        else:
            return f"{self.base_url}/{self.year}/acs/acs1/pums"

    def _make_request(
        self,
        variables: List[str],
        state_code: str,
        record_type: str = "person",
    ) -> List[List[str]]:
        """Make a request to the Census API.
        
        Args:
            variables: List of variable names to fetch.
            state_code: FIPS state code.
            record_type: Either 'person' or 'household'.
            
        Returns:
            List of records (each record is a list of values).
        """
        url = self._build_url(record_type)
        
        params = {
            "get": ",".join(variables),
            "for": f"state:{state_code}",
        }
        
        if self.api_key:
            params["key"] = self.api_key
        
        logger.info(f"Fetching {record_type} data for state {state_code}")
        
        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # First row is headers, rest is data
            if len(data) > 1:
                return data[1:]
            return []
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def fetch_person_data(self, state_code: str) -> List[Dict[str, Any]]:
        """Fetch person-level PUMS data for a state.
        
        Args:
            state_code: FIPS state code (e.g., '06' for California).
            
        Returns:
            List of person records as dictionaries.
        """
        raw_data = self._make_request(PERSON_VARIABLES, state_code, "person")
        
        # String variables that should not be converted to int
        string_vars = ["SERIALNO", "OCCP", "SOCP", "INDP", "NAICSP", "PUMA", "STATE", "FOD1P", "FOD2P"]
        
        records = []
        for row in raw_data:
            record = {}
            for i, var in enumerate(PERSON_VARIABLES):
                value = row[i] if i < len(row) else None
                
                # Map STATE to st for consistency
                db_var = "st" if var == "STATE" else var.lower()
                
                # Convert to appropriate type
                if value is not None and value != "":
                    if var in string_vars:
                        record[db_var] = str(value)
                    else:
                        try:
                            record[db_var] = int(value)
                        except (ValueError, TypeError):
                            record[db_var] = value
                else:
                    record[db_var] = None
            
            record["year"] = self.year
            records.append(record)
        
        logger.info(f"Fetched {len(records)} person records for state {state_code}")
        return records

    def fetch_household_data(self, state_code: str) -> List[Dict[str, Any]]:
        """Fetch household-level PUMS data for a state.
        
        Args:
            state_code: FIPS state code (e.g., '06' for California).
            
        Returns:
            List of household records as dictionaries.
        """
        raw_data = self._make_request(HOUSEHOLD_VARIABLES, state_code, "household")
        
        records = []
        for row in raw_data:
            record = {}
            for i, var in enumerate(HOUSEHOLD_VARIABLES):
                value = row[i] if i < len(row) else None
                # Map API variable names to database column names
                if var == "TYPEHUGQ":
                    db_var = "type_hu"
                elif var == "HUPAC":
                    db_var = "hupaoc"
                elif var == "STATE":
                    db_var = "st"
                else:
                    db_var = var.lower()
                
                if value is not None and value != "":
                    if var in ["SERIALNO", "PUMA", "STATE"]:
                        record[db_var] = str(value)
                    else:
                        try:
                            record[db_var] = int(value)
                        except (ValueError, TypeError):
                            record[db_var] = value
                else:
                    record[db_var] = None
            
            record["year"] = self.year
            records.append(record)
        
        logger.info(f"Fetched {len(records)} household records for state {state_code}")
        return records

    def get_available_states(self) -> Dict[str, str]:
        """Return dictionary of available state codes and names."""
        return STATE_CODES.copy()

    def test_connection(self) -> bool:
        """Test API connectivity.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Try to fetch a small amount of data
            url = self._build_url("person")
            params = {
                "get": "SERIALNO",
                "for": "state:01",  # Alabama
            }
            if self.api_key:
                params["key"] = self.api_key
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
