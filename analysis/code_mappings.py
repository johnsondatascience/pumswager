"""
ACS PUMS Code Mappings

This module provides descriptive labels for coded values in the ACS PUMS data.
Based on the 2023 ACS PUMS Data Dictionary.
"""

# State FIPS codes
STATE_CODES = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas',
    6: 'California', 8: 'Colorado', 9: 'Connecticut', 10: 'Delaware',
    11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii',
    16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa',
    20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine',
    24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota',
    28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska',
    32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico',
    36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio',
    40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island',
    45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 48: 'Texas',
    49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington',
    54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming', 72: 'Puerto Rico',
}

# Educational attainment (SCHL)
EDUCATION_CODES = {
    1: 'No schooling completed',
    2: 'Nursery school, preschool',
    3: 'Kindergarten',
    4: 'Grade 1',
    5: 'Grade 2',
    6: 'Grade 3',
    7: 'Grade 4',
    8: 'Grade 5',
    9: 'Grade 6',
    10: 'Grade 7',
    11: 'Grade 8',
    12: 'Grade 9',
    13: 'Grade 10',
    14: 'Grade 11',
    15: '12th grade - no diploma',
    16: 'Regular high school diploma',
    17: 'GED or alternative credential',
    18: 'Some college, but less than 1 year',
    19: '1 or more years of college credit, no degree',
    20: "Associate's degree",
    21: "Bachelor's degree",
    22: "Master's degree",
    23: 'Professional degree beyond bachelor\'s',
    24: 'Doctorate degree',
}

# Simplified education categories
EDUCATION_CATEGORIES = {
    1: 'Less than HS', 2: 'Less than HS', 3: 'Less than HS', 4: 'Less than HS',
    5: 'Less than HS', 6: 'Less than HS', 7: 'Less than HS', 8: 'Less than HS',
    9: 'Less than HS', 10: 'Less than HS', 11: 'Less than HS', 12: 'Less than HS',
    13: 'Less than HS', 14: 'Less than HS', 15: 'Less than HS',
    16: 'High School', 17: 'High School',
    18: 'Some College', 19: 'Some College',
    20: "Associate's",
    21: "Bachelor's",
    22: "Master's",
    23: 'Professional',
    24: 'Doctorate',
}

# Sex (SEX)
SEX_CODES = {
    1: 'Male',
    2: 'Female',
}

# Race (RAC1P)
RACE_CODES = {
    1: 'White alone',
    2: 'Black or African American alone',
    3: 'American Indian alone',
    4: 'Alaska Native alone',
    5: 'American Indian and Alaska Native tribes specified',
    6: 'Asian alone',
    7: 'Native Hawaiian and Other Pacific Islander alone',
    8: 'Some Other Race alone',
    9: 'Two or More Races',
}

# Hispanic origin (HISP)
HISPANIC_CODES = {
    1: 'Not Spanish/Hispanic/Latino',
    2: 'Mexican',
    3: 'Puerto Rican',
    4: 'Cuban',
    5: 'Dominican',
    6: 'Costa Rican',
    7: 'Guatemalan',
    8: 'Honduran',
    9: 'Nicaraguan',
    10: 'Panamanian',
    11: 'Salvadoran',
    12: 'Other Central American',
    13: 'Argentinean',
    14: 'Bolivian',
    15: 'Chilean',
    16: 'Colombian',
    17: 'Ecuadorian',
    18: 'Paraguayan',
    19: 'Peruvian',
    20: 'Uruguayan',
    21: 'Venezuelan',
    22: 'Other South American',
    23: 'Spaniard',
    24: 'All Other Spanish/Hispanic/Latino',
}

# Simplified Hispanic categories
def get_hispanic_category(code):
    if code == 1:
        return 'Not Hispanic'
    else:
        return 'Hispanic'

# Employment status recode (ESR)
EMPLOYMENT_STATUS_CODES = {
    1: 'Civilian employed, at work',
    2: 'Civilian employed, with a job but not at work',
    3: 'Unemployed',
    4: 'Armed forces, at work',
    5: 'Armed forces, with a job but not at work',
    6: 'Not in labor force',
}

# Class of worker (COW)
CLASS_OF_WORKER_CODES = {
    1: 'Private for-profit employee',
    2: 'Private not-for-profit employee',
    3: 'Local government employee',
    4: 'State government employee',
    5: 'Federal government employee',
    6: 'Self-employed (not incorporated)',
    7: 'Self-employed (incorporated)',
    8: 'Working without pay in family business',
    9: 'Unemployed, last worked 5+ years ago',
}

# Tenure (TEN)
TENURE_CODES = {
    1: 'Owned with mortgage/loan',
    2: 'Owned free and clear',
    3: 'Rented',
    4: 'Occupied without rent',
}

# Building type (BLD)
BUILDING_TYPE_CODES = {
    1: 'Mobile home or trailer',
    2: 'One-family house detached',
    3: 'One-family house attached',
    4: 'Apartment: 2 units',
    5: 'Apartment: 3-4 units',
    6: 'Apartment: 5-9 units',
    7: 'Apartment: 10-19 units',
    8: 'Apartment: 20-49 units',
    9: 'Apartment: 50+ units',
    10: 'Boat, RV, van, etc.',
}

# Household type (HHT)
HOUSEHOLD_TYPE_CODES = {
    1: 'Married couple household',
    2: 'Other family: Male householder, no spouse',
    3: 'Other family: Female householder, no spouse',
    4: 'Nonfamily: Male householder, living alone',
    5: 'Nonfamily: Male householder, not living alone',
    6: 'Nonfamily: Female householder, living alone',
    7: 'Nonfamily: Female householder, not living alone',
}

# Detailed household type (HHT2)
HOUSEHOLD_TYPE_DETAILED_CODES = {
    1: 'Married couple with children under 18',
    2: 'Married couple without children under 18',
    3: 'Cohabiting couple with children under 18',
    4: 'Cohabiting couple without children under 18',
    5: 'Female householder, no spouse/partner, with children under 18',
    6: 'Female householder, no spouse/partner, without children under 18',
    7: 'Male householder, no spouse/partner, with children under 18',
    8: 'Male householder, no spouse/partner, without children under 18',
}

# Type of housing unit (TYPEHUGQ)
HOUSING_UNIT_TYPE_CODES = {
    1: 'Housing unit',
    2: 'Institutional group quarters',
    3: 'Noninstitutional group quarters',
}

# Presence and age of children (HUPAC)
CHILDREN_PRESENCE_CODES = {
    1: 'With children under 6 years only',
    2: 'With children 6 to 17 years only',
    3: 'With children under 6 years and 6 to 17 years',
    4: 'No children',
}

# Means of transportation to work (JWTRNS)
TRANSPORTATION_CODES = {
    1: 'Car, truck, or van',
    2: 'Bus',
    3: 'Subway or elevated rail',
    4: 'Long-distance train or commuter rail',
    5: 'Light rail, streetcar, or trolley',
    6: 'Ferryboat',
    7: 'Taxicab',
    8: 'Motorcycle',
    9: 'Bicycle',
    10: 'Walked',
    11: 'Worked from home',
    12: 'Other method',
}


def map_codes(series, code_dict, default='Unknown'):
    """Map coded values to descriptive labels.
    
    Args:
        series: pandas Series with coded values
        code_dict: Dictionary mapping codes to labels
        default: Default value for unmapped codes
    
    Returns:
        Series with descriptive labels
    """
    return series.map(lambda x: code_dict.get(int(x), default) if pd.notna(x) else default)


def get_education_category(schl_value):
    """Get simplified education category from SCHL code."""
    if pd.isna(schl_value):
        return 'Unknown'
    return EDUCATION_CATEGORIES.get(int(schl_value), 'Unknown')


def estimate_education_years(schl_value):
    """Estimate years of education from SCHL code."""
    if pd.isna(schl_value):
        return None
    schl = int(schl_value)
    edu_years = {
        1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5,
        9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12,
        16: 12, 17: 12, 18: 13, 19: 14, 20: 14, 21: 16, 22: 18, 23: 20, 24: 22
    }
    return edu_years.get(schl, 12)


# Import pandas for type checking
import pandas as pd
