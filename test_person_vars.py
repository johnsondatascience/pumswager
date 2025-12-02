"""Test which person-level Census API variables are valid for 2023 ACS PUMS."""
import requests

API_KEY = "61dfcf7340d35d2e83a96ddb94a44f77c71e58a3"
BASE_URL = "https://api.census.gov/data/2023/acs/acs1/pums"

# New variables to test
PERSON_VARIABLES = [
    "SERIALNO", "SPORDER", "PUMA", "STATE",
    "AGEP", "SEX", "RAC1P", "HISP",
    "SCHL", "FOD1P", "FOD2P", "SCIENGP", "SCIENGRLP",
    "ESR", "COW", "OCCP", "SOCP", "NAICSP", "INDP",
    "WKHP", "WKW", "WKL", "JWMNP", "JWTRNS",
    "WAGP", "SEMP", "PINCP", "PERNP", "OIP", "INTP",
    "PWGTP",
]

def test_variable(var):
    """Test if a variable is valid."""
    params = {
        "get": var,
        "for": "state:06",
        "key": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        if response.status_code == 200:
            return True, "OK"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    results = ["Testing person variables for 2023 ACS PUMS...\n"]
    
    valid = []
    invalid = []
    
    for var in PERSON_VARIABLES:
        ok, msg = test_variable(var)
        status = "OK" if ok else "FAIL"
        results.append(f"  {status} {var}: {msg}")
        if ok:
            valid.append(var)
        else:
            invalid.append(var)
    
    results.append(f"\n{len(valid)} valid, {len(invalid)} invalid")
    if invalid:
        results.append(f"Invalid variables: {', '.join(invalid)}")
    
    with open("test_person_vars_output.txt", "w") as f:
        f.write("\n".join(results))
