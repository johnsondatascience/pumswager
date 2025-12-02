"""
ACS PUMS 2023 Codebook - Variable Labels and Value Mappings

This module provides meaningful labels for PUMS variable names and their coded values.
Based on the 2023 ACS PUMS Data Dictionary from the Census Bureau.
Source: https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2023.txt
"""

from typing import Dict, Optional, Any, Union

# =============================================================================
# VARIABLE LABELS - Descriptive names for each variable
# =============================================================================

PERSON_VARIABLE_LABELS: Dict[str, str] = {
    # Basic identifiers
    "serialno": "Housing Unit/GQ Person Serial Number",
    "sporder": "Person Number Within Household",
    "puma": "Public Use Microdata Area Code",
    "st": "State Code",
    "year": "Survey Year",
    
    # Demographics
    "agep": "Age",
    "sex": "Sex",
    "rac1p": "Race (Recoded)",
    "hisp": "Hispanic Origin (Detailed)",
    "nativity": "Nativity",
    "cit": "Citizenship Status",
    
    # Education
    "schl": "Educational Attainment",
    "fod1p": "Field of Degree - First Entry",
    "fod2p": "Field of Degree - Second Entry",
    "sciengp": "Science/Engineering Degree Flag",
    "sciengrlp": "Science/Engineering Related Degree Flag",
    
    # Employment & Occupation
    "esr": "Employment Status Recode",
    "cow": "Class of Worker",
    "occp": "Occupation Code (SOC-based)",
    "socp": "SOC Occupation Code (Detailed)",
    "naicsp": "NAICS Industry Code (Detailed)",
    "indp": "Industry Code",
    "wkhp": "Usual Hours Worked Per Week",
    "wkwn": "Weeks Worked Past 12 Months",
    "wkl": "When Last Worked",
    "jwmnp": "Travel Time to Work (Minutes)",
    "jwtrns": "Means of Transportation to Work",
    
    # Income & Earnings
    "wagp": "Wages/Salary Income (Past 12 Months)",
    "semp": "Self-Employment Income",
    "pincp": "Total Person Income",
    "pernp": "Total Person Earnings",
    "oip": "Other Income",
    "intp": "Interest/Dividends/Rental Income",
    "pap": "Public Assistance Income",
    "retp": "Retirement Income",
    "ssip": "Supplementary Security Income",
    "ssp": "Social Security Income",
    
    # Weight
    "pwgtp": "Person Weight",
}

HOUSEHOLD_VARIABLE_LABELS: Dict[str, str] = {
    # Basic identifiers
    "serialno": "Housing Unit Serial Number",
    "puma": "Public Use Microdata Area Code",
    "st": "State Code",
    "year": "Survey Year",
    
    # Housing characteristics
    "type_hu": "Type of Unit",
    "typehugq": "Type of Unit (Housing/Group Quarters)",
    "bld": "Units in Structure",
    "ten": "Tenure",
    "rmsp": "Number of Rooms",
    "bdsp": "Number of Bedrooms",
    "ybl": "Year Structure Built",
    
    # Household composition
    "np": "Number of Persons in Household",
    "hht": "Household/Family Type",
    "hht2": "Household/Family Type (Detailed)",
    "hupaoc": "Presence and Age of Own Children",
    "hupac": "Presence and Age of Children",
    "huparc": "Presence and Age of Related Children",
    
    # Income
    "hincp": "Household Income (Past 12 Months)",
    "fincp": "Family Income (Past 12 Months)",
    
    # Costs
    "grntp": "Gross Rent (Monthly)",
    "smocp": "Selected Monthly Owner Costs",
    "grpip": "Gross Rent as Percentage of Income",
    "ocpip": "Owner Costs as Percentage of Income",
    
    # Weight
    "wgtp": "Housing Unit Weight",
}


# =============================================================================
# VALUE LABELS - Meaningful labels for coded values
# =============================================================================

SEX_VALUES: Dict[int, str] = {
    1: "Male",
    2: "Female",
}

RAC1P_VALUES: Dict[int, str] = {
    1: "White alone",
    2: "Black or African American alone",
    3: "American Indian alone",
    4: "Alaska Native alone",
    5: "American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races",
    6: "Asian alone",
    7: "Native Hawaiian and Other Pacific Islander alone",
    8: "Some Other Race alone",
    9: "Two or More Races",
}

HISP_VALUES: Dict[int, str] = {
    1: "Not Spanish/Hispanic/Latino",
    2: "Mexican",
    3: "Puerto Rican",
    4: "Cuban",
    5: "Dominican",
    6: "Costa Rican",
    7: "Guatemalan",
    8: "Honduran",
    9: "Nicaraguan",
    10: "Panamanian",
    11: "Salvadoran",
    12: "Other Central American",
    13: "Argentinean",
    14: "Bolivian",
    15: "Chilean",
    16: "Colombian",
    17: "Ecuadorian",
    18: "Paraguayan",
    19: "Peruvian",
    20: "Uruguayan",
    21: "Venezuelan",
    22: "Other South American",
    23: "Spaniard",
    24: "All Other Spanish/Hispanic/Latino",
}

CIT_VALUES: Dict[int, str] = {
    1: "Born in the United States",
    2: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or Northern Marianas",
    3: "Born abroad of U.S. citizen parent or parents",
    4: "U.S. citizen by naturalization",
    5: "Not a U.S. citizen",
}

SCHL_VALUES: Dict[int, str] = {
    1: "No schooling completed",
    2: "Nursery school, preschool",
    3: "Kindergarten",
    4: "Grade 1",
    5: "Grade 2",
    6: "Grade 3",
    7: "Grade 4",
    8: "Grade 5",
    9: "Grade 6",
    10: "Grade 7",
    11: "Grade 8",
    12: "Grade 9",
    13: "Grade 10",
    14: "Grade 11",
    15: "12th grade - no diploma",
    16: "Regular high school diploma",
    17: "GED or alternative credential",
    18: "Some college, but less than 1 year",
    19: "1 or more years of college credit, no degree",
    20: "Associate's degree",
    21: "Bachelor's degree",
    22: "Master's degree",
    23: "Professional degree beyond a bachelor's degree",
    24: "Doctorate degree",
}

# Simplified education categories for analysis
EDUCATION_CATEGORIES: Dict[int, str] = {
    1: "Less than high school",
    2: "Less than high school",
    3: "Less than high school",
    4: "Less than high school",
    5: "Less than high school",
    6: "Less than high school",
    7: "Less than high school",
    8: "Less than high school",
    9: "Less than high school",
    10: "Less than high school",
    11: "Less than high school",
    12: "Less than high school",
    13: "Less than high school",
    14: "Less than high school",
    15: "Less than high school",
    16: "High school diploma",
    17: "High school diploma",
    18: "Some college",
    19: "Some college",
    20: "Associate's degree",
    21: "Bachelor's degree",
    22: "Master's degree",
    23: "Professional degree",
    24: "Doctorate degree",
}

ESR_VALUES: Dict[int, str] = {
    1: "Civilian employed, at work",
    2: "Civilian employed, with a job but not at work",
    3: "Unemployed",
    4: "Armed forces, at work",
    5: "Armed forces, with a job but not at work",
    6: "Not in labor force",
}

COW_VALUES: Dict[int, str] = {
    1: "Private for-profit employee",
    2: "Private not-for-profit employee",
    3: "Local government employee",
    4: "State government employee",
    5: "Federal government employee",
    6: "Self-employed (not incorporated)",
    7: "Self-employed (incorporated)",
    8: "Working without pay in family business or farm",
    9: "Unemployed, last worked 5+ years ago or never worked",
}

WKL_VALUES: Dict[int, str] = {
    1: "Within the past 12 months",
    2: "1-5 years ago",
    3: "Over 5 years ago or never worked",
}

JWTRNS_VALUES: Dict[int, str] = {
    1: "Car, truck, or van",
    2: "Bus",
    3: "Subway or elevated rail",
    4: "Long-distance train or commuter rail",
    5: "Light rail, streetcar, or trolley",
    6: "Ferryboat",
    7: "Taxicab",
    8: "Motorcycle",
    9: "Bicycle",
    10: "Walked",
    11: "Worked from home",
    12: "Other method",
}

# Household variables
TYPEHUGQ_VALUES: Dict[int, str] = {
    1: "Housing unit",
    2: "Institutional group quarters",
    3: "Noninstitutional group quarters",
}

BLD_VALUES: Dict[int, str] = {
    1: "Mobile home or trailer",
    2: "One-family house detached",
    3: "One-family house attached",
    4: "2 Apartments",
    5: "3-4 Apartments",
    6: "5-9 Apartments",
    7: "10-19 Apartments",
    8: "20-49 Apartments",
    9: "50 or more apartments",
    10: "Boat, RV, van, etc.",
}

TEN_VALUES: Dict[int, str] = {
    1: "Owned with mortgage or loan",
    2: "Owned free and clear",
    3: "Rented",
    4: "Occupied without payment of rent",
}

HHT_VALUES: Dict[int, str] = {
    1: "Married couple household",
    2: "Other family household: Male householder, no spouse present",
    3: "Other family household: Female householder, no spouse present",
    4: "Nonfamily household: Male householder: Living alone",
    5: "Nonfamily household: Male householder: Not living alone",
    6: "Nonfamily household: Female householder: Living alone",
    7: "Nonfamily household: Female householder: Not living alone",
}

HHT2_VALUES: Dict[int, str] = {
    1: "Married couple household with children under 18",
    2: "Married couple household, no children under 18",
    3: "Cohabiting couple household with children under 18",
    4: "Cohabiting couple household, no children under 18",
    5: "Female householder, no spouse/partner, living alone",
    6: "Female householder, no spouse/partner, with children under 18",
    7: "Female householder, no spouse/partner, with relatives, no children under 18",
    8: "Female householder, no spouse/partner, only nonrelatives present",
    9: "Male householder, no spouse/partner, living alone",
    10: "Male householder, no spouse/partner, with children under 18",
    11: "Male householder, no spouse/partner, with relatives, no children under 18",
    12: "Male householder, no spouse/partner, only nonrelatives present",
}

HUPAC_VALUES: Dict[int, str] = {
    1: "With children under 6 years only",
    2: "With children 6 to 17 years only",
    3: "With children under 6 years and 6 to 17 years",
    4: "No children",
}

# State codes
STATE_CODES: Dict[str, str] = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "72": "Puerto Rico",
}

# Field of Degree codes (FOD1P, FOD2P)
FIELD_OF_DEGREE_VALUES: Dict[str, str] = {
    "1100": "General Agriculture",
    "1101": "Agriculture Production And Management",
    "1102": "Agricultural Economics",
    "1103": "Animal Sciences",
    "1104": "Food Science",
    "1105": "Plant Science And Agronomy",
    "1106": "Soil Science",
    "1107": "Veterinary Medicine",
    "1199": "Miscellaneous Agriculture",
    "1301": "Environmental Science",
    "1302": "Forestry",
    "1303": "Natural Resources Management",
    "1401": "Architecture",
    "1501": "Area Ethnic And Civilization Studies",
    "1901": "Communications",
    "1902": "Journalism",
    "1903": "Mass Media",
    "1904": "Advertising And Public Relations",
    "2001": "Communication Technologies",
    "2100": "Computer And Information Systems",
    "2101": "Computer Programming And Data Processing",
    "2102": "Computer Science",
    "2105": "Information Sciences",
    "2106": "Computer Administration Management And Security",
    "2107": "Computer Networking And Telecommunications",
    "2201": "Cosmetology Services And Culinary Arts",
    "2300": "General Education",
    "2301": "Educational Administration And Supervision",
    "2303": "School Student Counseling",
    "2304": "Elementary Education",
    "2305": "Mathematics Teacher Education",
    "2306": "Physical And Health Education Teaching",
    "2307": "Early Childhood Education",
    "2308": "Science And Computer Teacher Education",
    "2309": "Secondary Teacher Education",
    "2310": "Special Needs Education",
    "2311": "Social Science Or History Teacher Education",
    "2312": "Teacher Education: Multiple Levels",
    "2313": "Language And Drama Education",
    "2314": "Art And Music Education",
    "2399": "Miscellaneous Education",
    "2400": "General Engineering",
    "2401": "Aerospace Engineering",
    "2402": "Biological Engineering",
    "2403": "Architectural Engineering",
    "2404": "Biomedical Engineering",
    "2405": "Chemical Engineering",
    "2406": "Civil Engineering",
    "2407": "Computer Engineering",
    "2408": "Electrical Engineering",
    "2409": "Engineering Mechanics Physics And Science",
    "2410": "Environmental Engineering",
    "2411": "Geological And Geophysical Engineering",
    "2412": "Industrial And Manufacturing Engineering",
    "2413": "Materials Engineering And Materials Science",
    "2414": "Mechanical Engineering",
    "2415": "Metallurgical Engineering",
    "2416": "Mining And Mineral Engineering",
    "2417": "Naval Architecture And Marine Engineering",
    "2418": "Nuclear Engineering",
    "2419": "Petroleum Engineering",
    "2499": "Miscellaneous Engineering",
    "2500": "Engineering Technologies",
    "2501": "Engineering And Industrial Management",
    "2502": "Electrical Engineering Technology",
    "2503": "Industrial Production Technologies",
    "2504": "Mechanical Engineering Related Technologies",
    "2599": "Miscellaneous Engineering Technologies",
    "2601": "Linguistics And Comparative Language And Literature",
    "2602": "French German Latin And Other Common Foreign Language Studies",
    "2603": "Other Foreign Languages",
    "2901": "Family And Consumer Sciences",
    "3202": "Pre-Law And Legal Studies",
    "3301": "English Language And Literature",
    "3302": "Composition And Rhetoric",
    "3401": "Liberal Arts",
    "3402": "Humanities",
    "3501": "Library Science",
    "3600": "Biology",
    "3601": "Biochemical Sciences",
    "3602": "Botany",
    "3603": "Molecular Biology",
    "3604": "Ecology",
    "3605": "Genetics",
    "3606": "Microbiology",
    "3607": "Pharmacology",
    "3608": "Physiology",
    "3609": "Zoology",
    "3611": "Neuroscience",
    "3699": "Miscellaneous Biology",
    "3700": "Mathematics",
    "3701": "Applied Mathematics",
    "3702": "Statistics And Decision Science",
    "3801": "Military Technologies",
    "4000": "Multi/Interdisciplinary Studies",
    "4001": "Intercultural And International Studies",
    "4002": "Nutrition Sciences",
    "4005": "Mathematics And Computer Science",
    "4006": "Cognitive Science And Biopsychology",
    "4007": "Interdisciplinary Social Sciences",
    "4009": "Data Science and Data Analytics",
    "4101": "Physical Fitness Parks Recreation And Leisure",
    "4801": "Philosophy And Religious Studies",
    "4901": "Theology And Religious Vocations",
    "5000": "Physical Sciences",
    "5001": "Astronomy And Astrophysics",
    "5002": "Atmospheric Sciences And Meteorology",
    "5003": "Chemistry",
    "5004": "Geology And Earth Science",
    "5005": "Geosciences",
    "5006": "Oceanography",
    "5007": "Physics",
    "5008": "Materials Science",
    "5098": "Multi-Disciplinary Or General Science",
    "5102": "Nuclear, Industrial Radiology, And Biological Technologies",
    "5200": "Psychology",
    "5201": "Educational Psychology",
    "5202": "Clinical Psychology",
    "5203": "Counseling Psychology",
    "5205": "Industrial And Organizational Psychology",
    "5206": "Social Psychology",
    "5299": "Miscellaneous Psychology",
    "5301": "Criminal Justice And Fire Protection",
    "5401": "Public Administration",
    "5402": "Public Policy",
    "5403": "Human Services And Community Organization",
    "5404": "Social Work",
    "5500": "General Social Sciences",
    "5501": "Economics",
    "5502": "Anthropology And Archeology",
    "5503": "Criminology",
    "5504": "Geography",
    "5505": "International Relations",
    "5506": "Political Science And Government",
    "5507": "Sociology",
    "5599": "Miscellaneous Social Sciences",
    "5601": "Construction Services",
    "5701": "Electrical, Mechanical, And Precision Technologies And Production",
    "5901": "Transportation Sciences And Technologies",
    "6000": "Fine Arts",
    "6001": "Drama And Theater Arts",
    "6002": "Music",
    "6003": "Visual And Performing Arts",
    "6004": "Commercial Art And Graphic Design",
    "6005": "Film Video And Photographic Arts",
    "6006": "Art History And Criticism",
    "6007": "Studio Arts",
    "6099": "Miscellaneous Fine Arts",
    "6100": "General Medical And Health Services",
    "6102": "Communication Disorders Sciences And Services",
    "6103": "Health And Medical Administrative Services",
    "6104": "Medical Assisting Services",
    "6105": "Medical Technologies Technicians",
    "6106": "Health And Medical Preparatory Programs",
    "6107": "Nursing",
    "6108": "Pharmacy Pharmaceutical Sciences And Administration",
    "6109": "Treatment Therapy Professions",
    "6110": "Community And Public Health",
    "6199": "Miscellaneous Health Medical Professions",
    "6200": "General Business",
    "6201": "Accounting",
    "6202": "Actuarial Science",
    "6203": "Business Management And Administration",
    "6204": "Operations Logistics And E-Commerce",
    "6205": "Business Economics",
    "6206": "Marketing And Marketing Research",
    "6207": "Finance",
    "6209": "Human Resources And Personnel Management",
    "6210": "International Business",
    "6211": "Hospitality Management",
    "6212": "Management Information Systems And Statistics",
    "6299": "Miscellaneous Business And Medical Administration",
    "6402": "History",
    "6403": "United States History",
}


# =============================================================================
# OCCUPATION CATEGORIES (Major Groups)
# =============================================================================

OCCUPATION_MAJOR_GROUPS: Dict[str, str] = {
    "MGR": "Management Occupations",
    "BUS": "Business and Financial Operations Occupations",
    "CMM": "Computer and Mathematical Occupations",
    "ENG": "Architecture and Engineering Occupations",
    "SCI": "Life, Physical, and Social Science Occupations",
    "CMS": "Community and Social Service Occupations",
    "LGL": "Legal Occupations",
    "EDU": "Educational Instruction and Library Occupations",
    "ENT": "Arts, Design, Entertainment, Sports, and Media Occupations",
    "MED": "Healthcare Practitioners and Technical Occupations",
    "HLS": "Healthcare Support Occupations",
    "PRT": "Protective Service Occupations",
    "EAT": "Food Preparation and Serving Related Occupations",
    "CLN": "Building and Grounds Cleaning and Maintenance Occupations",
    "PRS": "Personal Care and Service Occupations",
    "SAL": "Sales and Related Occupations",
    "OFF": "Office and Administrative Support Occupations",
    "FFF": "Farming, Fishing, and Forestry Occupations",
    "CON": "Construction and Extraction Occupations",
    "EXT": "Extraction Occupations",
    "RPR": "Installation, Maintenance, and Repair Occupations",
    "PRD": "Production Occupations",
    "TRN": "Transportation and Material Moving Occupations",
    "MIL": "Military Specific Occupations",
}

# =============================================================================
# INDUSTRY CATEGORIES (Major Groups)
# =============================================================================

INDUSTRY_MAJOR_GROUPS: Dict[str, str] = {
    "AGR": "Agriculture, Forestry, Fishing and Hunting",
    "EXT": "Mining, Quarrying, and Oil and Gas Extraction",
    "UTL": "Utilities",
    "CON": "Construction",
    "MFG": "Manufacturing",
    "WHL": "Wholesale Trade",
    "RET": "Retail Trade",
    "TRN": "Transportation and Warehousing",
    "INF": "Information",
    "FIN": "Finance and Insurance, and Real Estate and Rental and Leasing",
    "PRF": "Professional, Scientific, and Management, and Administrative and Waste Management Services",
    "EDU": "Educational Services",
    "MED": "Health Care and Social Assistance",
    "SCA": "Social Assistance",
    "ENT": "Arts, Entertainment, and Recreation, and Accommodation and Food Services",
    "SRV": "Other Services, Except Public Administration",
    "ADM": "Public Administration",
    "MIL": "Military",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_variable_label(variable: str, record_type: str = "person") -> str:
    """
    Get the descriptive label for a variable.
    
    Args:
        variable: Variable name (e.g., 'agep', 'sex')
        record_type: Either 'person' or 'household'
        
    Returns:
        Descriptive label for the variable
    """
    var_lower = variable.lower()
    
    if record_type == "person":
        return PERSON_VARIABLE_LABELS.get(var_lower, variable.upper())
    else:
        return HOUSEHOLD_VARIABLE_LABELS.get(var_lower, variable.upper())


def get_value_label(variable: str, value: Union[int, str, None]) -> str:
    """
    Get the descriptive label for a coded value.
    
    Args:
        variable: Variable name (e.g., 'sex', 'rac1p')
        value: The coded value
        
    Returns:
        Descriptive label for the value, or the original value if not found
    """
    if value is None:
        return "N/A"
    
    var_lower = variable.lower()
    
    # Map variable names to their value dictionaries
    value_maps = {
        "sex": SEX_VALUES,
        "rac1p": RAC1P_VALUES,
        "hisp": HISP_VALUES,
        "cit": CIT_VALUES,
        "schl": SCHL_VALUES,
        "esr": ESR_VALUES,
        "cow": COW_VALUES,
        "wkl": WKL_VALUES,
        "jwtrns": JWTRNS_VALUES,
        "typehugq": TYPEHUGQ_VALUES,
        "type_hu": TYPEHUGQ_VALUES,
        "bld": BLD_VALUES,
        "ten": TEN_VALUES,
        "hht": HHT_VALUES,
        "hht2": HHT2_VALUES,
        "hupac": HUPAC_VALUES,
        "hupaoc": HUPAC_VALUES,
        "huparc": HUPAC_VALUES,
        "st": STATE_CODES,
        "fod1p": FIELD_OF_DEGREE_VALUES,
        "fod2p": FIELD_OF_DEGREE_VALUES,
    }
    
    if var_lower in value_maps:
        value_dict = value_maps[var_lower]
        # Handle both int and string keys
        if isinstance(value, int) and value in value_dict:
            return value_dict[value]
        elif str(value) in value_dict:
            return value_dict[str(value)]
        elif str(value).zfill(2) in value_dict:
            return value_dict[str(value).zfill(2)]
    
    return str(value)


def get_education_category(schl_value: Optional[int]) -> str:
    """
    Get a simplified education category from SCHL value.
    
    Args:
        schl_value: Educational attainment code (1-24)
        
    Returns:
        Simplified education category
    """
    if schl_value is None:
        return "N/A"
    return EDUCATION_CATEGORIES.get(schl_value, "Unknown")


def get_state_name(state_code: str) -> str:
    """
    Get state name from FIPS code.
    
    Args:
        state_code: Two-digit FIPS state code
        
    Returns:
        State name
    """
    return STATE_CODES.get(str(state_code).zfill(2), f"Unknown ({state_code})")


def get_occupation_group(occp_code: str) -> str:
    """
    Get the major occupation group from an occupation code.
    
    Args:
        occp_code: Occupation code (e.g., '1000', '2100')
        
    Returns:
        Major occupation group name
    """
    if not occp_code or occp_code == "":
        return "N/A"
    
    # Extract the prefix from the occupation code
    # The first 1-4 digits typically indicate the major group
    code_str = str(occp_code).strip()
    
    # Map numeric ranges to occupation groups
    try:
        code_num = int(code_str)
        if 10 <= code_num < 1000:
            return OCCUPATION_MAJOR_GROUPS.get("MGR", "Management")
        elif 1000 <= code_num < 1100:
            return OCCUPATION_MAJOR_GROUPS.get("BUS", "Business and Financial")
        elif 1000 <= code_num < 1300:
            return OCCUPATION_MAJOR_GROUPS.get("CMM", "Computer and Mathematical")
        elif 1300 <= code_num < 1600:
            return OCCUPATION_MAJOR_GROUPS.get("ENG", "Architecture and Engineering")
        elif 1600 <= code_num < 2000:
            return OCCUPATION_MAJOR_GROUPS.get("SCI", "Life, Physical, and Social Science")
        elif 2000 <= code_num < 2100:
            return OCCUPATION_MAJOR_GROUPS.get("CMS", "Community and Social Service")
        elif 2100 <= code_num < 2200:
            return OCCUPATION_MAJOR_GROUPS.get("LGL", "Legal")
        elif 2200 <= code_num < 2600:
            return OCCUPATION_MAJOR_GROUPS.get("EDU", "Education")
        elif 2600 <= code_num < 3000:
            return OCCUPATION_MAJOR_GROUPS.get("ENT", "Arts, Entertainment, Sports, Media")
        elif 3000 <= code_num < 3600:
            return OCCUPATION_MAJOR_GROUPS.get("MED", "Healthcare Practitioners")
        elif 3600 <= code_num < 3700:
            return OCCUPATION_MAJOR_GROUPS.get("HLS", "Healthcare Support")
        elif 3700 <= code_num < 4000:
            return OCCUPATION_MAJOR_GROUPS.get("PRT", "Protective Service")
        elif 4000 <= code_num < 4200:
            return OCCUPATION_MAJOR_GROUPS.get("EAT", "Food Preparation and Serving")
        elif 4200 <= code_num < 4300:
            return OCCUPATION_MAJOR_GROUPS.get("CLN", "Building and Grounds Cleaning")
        elif 4300 <= code_num < 4700:
            return OCCUPATION_MAJOR_GROUPS.get("PRS", "Personal Care and Service")
        elif 4700 <= code_num < 5000:
            return OCCUPATION_MAJOR_GROUPS.get("SAL", "Sales")
        elif 5000 <= code_num < 6000:
            return OCCUPATION_MAJOR_GROUPS.get("OFF", "Office and Administrative Support")
        elif 6000 <= code_num < 6200:
            return OCCUPATION_MAJOR_GROUPS.get("FFF", "Farming, Fishing, and Forestry")
        elif 6200 <= code_num < 7000:
            return OCCUPATION_MAJOR_GROUPS.get("CON", "Construction and Extraction")
        elif 7000 <= code_num < 7700:
            return OCCUPATION_MAJOR_GROUPS.get("RPR", "Installation, Maintenance, Repair")
        elif 7700 <= code_num < 9000:
            return OCCUPATION_MAJOR_GROUPS.get("PRD", "Production")
        elif 9000 <= code_num < 9800:
            return OCCUPATION_MAJOR_GROUPS.get("TRN", "Transportation and Material Moving")
        elif 9800 <= code_num < 9920:
            return OCCUPATION_MAJOR_GROUPS.get("MIL", "Military")
    except (ValueError, TypeError):
        pass
    
    return "Unknown"


def decode_record(record: Dict[str, Any], record_type: str = "person") -> Dict[str, Any]:
    """
    Decode a record by adding descriptive labels for all coded values.
    
    Args:
        record: Dictionary containing PUMS data
        record_type: Either 'person' or 'household'
        
    Returns:
        New dictionary with added '_label' fields for coded values
    """
    decoded = record.copy()
    
    # Variables that have value labels
    labeled_vars = ["sex", "rac1p", "hisp", "cit", "schl", "esr", "cow", "wkl", 
                    "jwtrns", "typehugq", "type_hu", "bld", "ten", "hht", "hht2",
                    "hupac", "hupaoc", "huparc", "st", "fod1p", "fod2p"]
    
    for var in labeled_vars:
        if var in record and record[var] is not None:
            decoded[f"{var}_label"] = get_value_label(var, record[var])
    
    # Add education category if SCHL is present
    if "schl" in record and record["schl"] is not None:
        decoded["education_category"] = get_education_category(record["schl"])
    
    # Add state name if ST is present
    if "st" in record and record["st"] is not None:
        decoded["state_name"] = get_state_name(record["st"])
    
    # Add occupation group if OCCP is present
    if "occp" in record and record["occp"] is not None:
        decoded["occupation_group"] = get_occupation_group(record["occp"])
    
    return decoded


# =============================================================================
# METADATA
# =============================================================================

CODEBOOK_METADATA = {
    "source": "2023 ACS PUMS Data Dictionary",
    "source_url": "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2023.txt",
    "documentation_url": "https://www.census.gov/programs-surveys/acs/microdata/documentation.html",
    "last_updated": "October 17, 2024",
    "survey_year": 2023,
}


if __name__ == "__main__":
    # Example usage
    print("ACS PUMS 2023 Codebook")
    print("=" * 50)
    
    # Example: Get variable labels
    print("\nPerson Variable Labels:")
    for var, label in list(PERSON_VARIABLE_LABELS.items())[:5]:
        print(f"  {var}: {label}")
    
    # Example: Get value labels
    print("\nSex Values:")
    for code, label in SEX_VALUES.items():
        print(f"  {code}: {label}")
    
    print("\nRace Values:")
    for code, label in RAC1P_VALUES.items():
        print(f"  {code}: {label}")
    
    print("\nEducation Values:")
    for code, label in list(SCHL_VALUES.items())[-5:]:
        print(f"  {code}: {label}")
    
    # Example: Decode a sample record
    sample_record = {
        "sex": 1,
        "rac1p": 6,
        "schl": 21,
        "esr": 1,
        "st": "06",
    }
    
    print("\nSample Record Decoding:")
    decoded = decode_record(sample_record)
    for key, value in decoded.items():
        print(f"  {key}: {value}")
