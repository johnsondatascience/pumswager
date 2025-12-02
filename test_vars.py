"""Test which Census API variables are valid for 2023 ACS PUMS."""
import requests

API_KEY = "61dfcf7340d35d2e83a96ddb94a44f77c71e58a3"
BASE_URL = "https://api.census.gov/data/2023/acs/acs1/pums"

# Variables to test
HOUSEHOLD_VARIABLES = [
    "SERIALNO",  # Serial number
    "PUMA",      # Public Use Microdata Area
    "ST",        # State code
    "TYPE",      # Type of unit
    "BLD",       # Building type
    "TEN",       # Tenure
    "RMSP",      # Rooms
    "BDSP",      # Bedrooms
    "YBL",       # Year built
    "NP",        # Number of persons
    "HHT",       # Household type
    "HUPAOC",    # Children present
    "HUPARC",    # Related children
    "HINCP",     # Household income
    "FINCP",     # Family income
    "GRNTP",     # Gross rent
    "SMOCP",     # Owner costs
    "GRPIP",     # Rent as % income
    "OCPIP",     # Owner costs as % income
    "WGTP",      # Housing weight
]

def test_variable(var):
    """Test if a variable is valid."""
    url = BASE_URL
    params = {
        "get": var,
        "for": "state:01",
        "key": API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return True, "OK"
        else:
            return False, f"Status {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return False, str(e)

results = []
results.append("Testing household variables for 2023 ACS PUMS API...")
results.append("-" * 60)

for var in HOUSEHOLD_VARIABLES:
    valid, msg = test_variable(var)
    status = "OK" if valid else "FAIL"
    results.append(f"{status} {var}: {msg if not valid else 'Valid'}")

# Write to file
with open("test_results.txt", "w") as f:
    f.write("\n".join(results))

print("Results written to test_results.txt")
