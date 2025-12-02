"""Quick test of household data collection."""
import sys
import traceback

# Write to file immediately to confirm script starts
with open("run_test_output.txt", "w") as f:
    f.write("Script started\n")

try:
    from src.census_api import CensusAPIClient, HOUSEHOLD_VARIABLES
    from src.config import census_config
except Exception as e:
    with open("run_test_output.txt", "a") as f:
        f.write(f"Import error: {e}\n")
        f.write(traceback.format_exc())
    raise

print("Testing Census API household variables...")
print(f"Year: {census_config.year}")
print(f"Variables: {HOUSEHOLD_VARIABLES}")

client = CensusAPIClient()

# Test with California (06)
print("\nTesting API request for California (state 06)...")
try:
    data = client.fetch_household_data("06")
    print(f"SUCCESS! Fetched {len(data)} records")
    if data:
        print(f"Sample record keys: {list(data[0].keys())}")
except Exception as e:
    print(f"ERROR: {e}")

# Write results to file
with open("run_test_output.txt", "w") as f:
    f.write("Test completed\n")
