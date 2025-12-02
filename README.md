# ACS PUMS Data Collector

A Python application that collects American Community Survey (ACS) Public Use Microdata Sample (PUMS) data from the Census Bureau API and stores it in a PostgreSQL database.

## Features

- **Dockerized PostgreSQL database** for data storage
- **Census API integration** for fetching PUMS data
- **Person-level data collection** (demographics, employment, income)
- **Household-level data collection** (housing characteristics, costs)
- **Job tracking** for monitoring collection progress
- **Upsert support** to handle duplicate records gracefully

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Census API key (free, get one at https://api.census.gov/data/key_signup.html)

## Quick Start

### 1. Clone and Setup

```bash
cd pumswager

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac

# Edit .env and add your Census API key
```

### 3. Start Database

```bash
docker-compose up -d
```

The database will be initialized automatically with the required schema.

### 4. Verify Setup

```bash
python main.py --check
```

### 5. Collect Data

```bash
# Collect all data for California (state code 06)
python main.py --collect all --states 06

# Collect person data for multiple states
python main.py --collect person --states 06 36 48

# Collect household data for all states (takes a while!)
python main.py --collect household

# Specify a different year
python main.py --collect all --year 2021 --states 06
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `--check` | Verify database and API connectivity |
| `--list-states` | Show available FIPS state codes |
| `--collect person` | Collect person-level data |
| `--collect household` | Collect household-level data |
| `--collect all` | Collect both person and household data |
| `--states CODE [CODE ...]` | Specify state codes (default: all states) |
| `--year YEAR` | Specify survey year (default: 2022) |

## Database Schema

### `pums_person` - Person-level records
- Demographics: age, sex, race, Hispanic origin, citizenship
- Education: educational attainment
- Employment: status, class of worker, occupation, industry
- Income: wages, self-employment, total income, benefits
- Weight: person weight for statistical analysis

### `pums_household` - Household-level records
- Housing: type, building, tenure, rooms, year built
- Composition: number of persons, household type, children
- Income: household and family income
- Costs: rent, owner costs, cost burden percentages
- Weight: housing unit weight for statistical analysis

### `collection_jobs` - Job tracking
- Tracks status of each collection job
- Records count and any errors

## Project Structure

```
pumswager/
├── docker-compose.yml      # Docker configuration
├── init_db/
│   └── 01_create_schema.sql  # Database initialization
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── database.py         # Database connection
│   ├── models.py           # SQLAlchemy ORM models
│   ├── census_api.py       # Census API client
│   └── collector.py        # Data collection orchestrator
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_HOST` | Database host | `localhost` |
| `POSTGRES_PORT` | Database port | `5432` |
| `POSTGRES_USER` | Database user | `pums_user` |
| `POSTGRES_PASSWORD` | Database password | `pums_password` |
| `POSTGRES_DB` | Database name | `pums_db` |
| `CENSUS_API_KEY` | Census Bureau API key | (required) |
| `ACS_YEAR` | Default survey year | `2022` |

## Data Variables Collected

### Person Variables
`SERIALNO`, `SPORDER`, `PUMA`, `ST`, `AGEP`, `SEX`, `RAC1P`, `HISP`, `NATIVITY`, `CIT`, `SCHL`, `ESR`, `COW`, `OCCP`, `INDP`, `WKHP`, `WKW`, `WAGP`, `SEMP`, `PINCP`, `PAP`, `RETP`, `SSIP`, `SSP`, `PWGTP`

### Household Variables
`SERIALNO`, `PUMA`, `ST`, `TYPE`, `BLD`, `TEN`, `RMSP`, `BDSP`, `YBL`, `NP`, `HHT`, `HUPAOC`, `HUPARC`, `HINCP`, `FINCP`, `GRNTP`, `SMOCP`, `GRPIP`, `OCPIP`, `WGTP`

## Codebook Module

The project includes a comprehensive codebook module (`src/codebook.py`) that provides meaningful labels for PUMS variable names and coded values. This is based on the official [2023 ACS PUMS Data Dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2023.txt).

### Usage

```python
from src.codebook import (
    get_variable_label,
    get_value_label,
    get_state_name,
    get_education_category,
    decode_record,
    SEX_VALUES,
    RAC1P_VALUES,
    SCHL_VALUES,
)

# Get descriptive label for a variable
print(get_variable_label('agep'))  # "Age"
print(get_variable_label('hincp', 'household'))  # "Household Income (Past 12 Months)"

# Get label for a coded value
print(get_value_label('sex', 1))  # "Male"
print(get_value_label('rac1p', 6))  # "Asian alone"
print(get_value_label('schl', 21))  # "Bachelor's degree"

# Get state name from FIPS code
print(get_state_name('06'))  # "California"

# Get simplified education category
print(get_education_category(21))  # "Bachelor's degree"

# Decode an entire record with labels
record = {'sex': 1, 'rac1p': 6, 'schl': 21, 'st': '06'}
decoded = decode_record(record)
# decoded now includes: sex_label, rac1p_label, schl_label, state_name, education_category
```

### Available Value Mappings

| Variable | Description |
|----------|-------------|
| `SEX_VALUES` | Sex (Male/Female) |
| `RAC1P_VALUES` | Race categories |
| `HISP_VALUES` | Hispanic origin (detailed) |
| `CIT_VALUES` | Citizenship status |
| `SCHL_VALUES` | Educational attainment (24 levels) |
| `ESR_VALUES` | Employment status |
| `COW_VALUES` | Class of worker |
| `TEN_VALUES` | Tenure (own/rent) |
| `HHT_VALUES` | Household type |
| `BLD_VALUES` | Building type |
| `STATE_CODES` | FIPS state codes |
| `FIELD_OF_DEGREE_VALUES` | Field of degree codes |

## API Rate Limits

The Census API has rate limits. The application includes:
- Automatic retry with exponential backoff
- 120-second timeout for large requests
- Batch inserts to minimize database load

## Wage Prediction Web Application

An interactive web service that estimates annual wages based on user-provided characteristics.

### Features

- **Interactive Form**: Input location, age, gender, education, occupation, and experience
- **ML-Based Predictions**: Uses Gradient Boosting trained on PUMS data
- **Confidence Intervals**: Shows estimated wage range
- **Survey Weights**: Model training respects Census survey weights
- **Subset Training**: Train models on filtered data subsets

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (requires data/pums_person_2023.csv)
python run_webapp.py --train-only

# Start the web server
python run_webapp.py

# Or train and start in one command
python run_webapp.py --train
```

The web app will be available at http://127.0.0.1:8000

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/health` | GET | Health check |
| `/api/options` | GET | Get form options (states, education levels, etc.) |
| `/api/predict` | POST | Predict wage for given characteristics |
| `/api/model/info` | GET | Get current model information |
| `/api/model/train` | POST | Train a new model |
| `/api/model/load` | POST | Load a saved model |
| `/docs` | GET | Interactive API documentation |

### Prediction Request Example

```python
import requests

response = requests.post("http://127.0.0.1:8000/api/predict", json={
    "state": "06",           # California
    "age": 35,
    "sex": 1,                # Male
    "education_level": "Bachelor's degree",
    "occupation_code": "15", # Computer & Mathematical
    "hours_per_week": 40,
    "class_of_worker": 1     # Private for-profit
})

result = response.json()
print(f"Predicted wage: {result['formatted_wage']}")
print(f"Range: {result['formatted_range']}")
```

### Training Custom Models

Train models on data subsets for more targeted predictions:

```python
import requests

# Train model for California only
response = requests.post("http://127.0.0.1:8000/api/model/train", json={
    "model_name": "california",
    "states": ["06"],
    "model_type": "gradient_boosting",
    "use_sample_weights": True
})
```

### Model Performance

The default Gradient Boosting model achieves:
- **R² Score**: ~0.45 (explains 45% of wage variation)
- **MAE**: ~$33,000 (average prediction error)
- **Sample Size**: ~1.2 million workers

Key predictive features:
1. Hours worked per week
2. Occupation category
3. Education level
4. Experience
5. State/location

## License

MIT
