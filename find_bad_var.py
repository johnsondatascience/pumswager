"""Find which variables are invalid for 2023 ACS PUMS API."""
import requests

API_KEY = "61dfcf7340d35d2e83a96ddb94a44f77c71e58a3"
BASE_URL = "https://api.census.gov/data/2023/acs/acs1/pums"

VARIABLES = [
    "SERIALNO", "PUMA", "ST", "TYPEHUGQ", "BLD", "TEN", "NP", 
    "HHT", "HHT2", "HUPAC", "HINCP", "FINCP", "GRNTP", "SMOCP", 
    "GRPIP", "OCPIP", "WGTP"
]

print("Testing each variable individually...")
print("-" * 50)

invalid_vars = []
valid_vars = []

for var in VARIABLES:
    url = f"{BASE_URL}?get={var}&for=state:01&key={API_KEY}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            valid_vars.append(var)
            print(f"OK: {var}")
        else:
            invalid_vars.append(var)
            print(f"FAIL: {var} - Status {r.status_code}")
    except Exception as e:
        invalid_vars.append(var)
        print(f"ERROR: {var} - {e}")

print("\n" + "=" * 50)
print(f"Valid variables ({len(valid_vars)}): {valid_vars}")
print(f"Invalid variables ({len(invalid_vars)}): {invalid_vars}")
