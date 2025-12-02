"""Download the ACS PUMS 2023 Data Dictionary from Census Bureau."""

import os
import requests

def download_data_dictionary():
    """Download the PUMS data dictionary to the data folder."""
    url = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2023.txt"
    output_path = os.path.join("data", "PUMS_Data_Dictionary_2023.txt")
    
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Successfully downloaded {len(response.text):,} characters")
        print(f"File size: {os.path.getsize(output_path):,} bytes")
        
    except Exception as e:
        print(f"Error downloading: {e}")
        return False
    
    return True


if __name__ == "__main__":
    download_data_dictionary()
