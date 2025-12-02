"""Find invalid person variables by testing each one."""
import requests
import sys

API_KEY = "61dfcf7340d35d2e83a96ddb94a44f77c71e58a3"
BASE_URL = "https://api.census.gov/data/2023/acs/acs1/pums"

PERSON_VARIABLES = [
    "SERIALNO", "SPORDER", "PUMA", "STATE",
    "AGEP", "SEX", "RAC1P", "HISP",
    "SCHL", "FOD1P", "FOD2P", "SCIENGP", "SCIENGRLP",
    "ESR", "COW", "OCCP", "SOCP", "NAICSP", "INDP",
    "WKHP", "WKW", "WKL", "JWMNP", "JWTRNS",
    "WAGP", "SEMP", "PINCP", "PERNP", "OIP", "INTP",
    "PWGTP",
]

valid = []
invalid = []

for var in PERSON_VARIABLES:
    params = {"get": var, "for": "state:01", "key": API_KEY}
    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        if r.status_code == 200:
            valid.append(var)
            sys.stdout.write(f"OK: {var}\n")
        else:
            invalid.append(var)
            sys.stdout.write(f"FAIL: {var} (HTTP {r.status_code})\n")
        sys.stdout.flush()
    except Exception as e:
        invalid.append(var)
        sys.stdout.write(f"FAIL: {var} ({e})\n")
        sys.stdout.flush()

sys.stdout.write(f"\nValid: {len(valid)}, Invalid: {len(invalid)}\n")
if invalid:
    sys.stdout.write(f"Invalid: {', '.join(invalid)}\n")
sys.stdout.flush()
