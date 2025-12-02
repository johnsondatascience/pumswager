"""
PUMS Wage Estimator - Dash Web Application

A simple interactive web app for estimating annual wages based on 
demographic and employment characteristics using Census PUMS data.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebook import (
    STATE_CODES,
    SEX_VALUES,
    COW_VALUES,
    SCHL_VALUES,
    get_education_category,
)

# ============================================================================
# Configuration
# ============================================================================

MODELS_DIR = Path(__file__).parent.parent / 'models'
DATA_DIR = Path(__file__).parent.parent / 'data'

# Education category to years mapping
EDUCATION_CATEGORY_YEARS = {
    'Less than high school': 10,
    'High school diploma': 12,
    'Some college': 14,
    "Associate's degree": 14,
    "Bachelor's degree": 16,
    "Master's degree": 18,
    'Professional degree': 20,
    'Doctorate degree': 21
}

EDUCATION_LEVELS = list(EDUCATION_CATEGORY_YEARS.keys())

# Occupation categories
OCCUPATION_CATEGORIES = {
    '': 'Any Occupation',
    '11': 'Management',
    '13': 'Business & Financial',
    '15': 'Computer & Mathematical',
    '17': 'Architecture & Engineering',
    '19': 'Life, Physical & Social Science',
    '21': 'Community & Social Service',
    '23': 'Legal',
    '25': 'Education & Library',
    '27': 'Arts, Entertainment & Media',
    '29': 'Healthcare Practitioners',
    '31': 'Healthcare Support',
    '33': 'Protective Service',
    '35': 'Food Preparation & Serving',
    '37': 'Building & Grounds Maintenance',
    '39': 'Personal Care & Service',
    '41': 'Sales',
    '43': 'Office & Administrative',
    '45': 'Farming, Fishing & Forestry',
    '47': 'Construction & Extraction',
    '49': 'Installation & Maintenance',
    '51': 'Production',
    '53': 'Transportation & Material Moving',
}

# ============================================================================
# Model Loading
# ============================================================================

model = None
preprocessor = None
model_metadata = None


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, model_metadata
    
    model_path = MODELS_DIR / 'wage_model_default.joblib'
    preprocessor_path = MODELS_DIR / 'wage_preprocessor_default.joblib'
    metadata_path = MODELS_DIR / 'wage_metadata_default.joblib'
    
    # Try legacy paths
    if not model_path.exists():
        model_path = MODELS_DIR / 'wage_prediction_model.joblib'
        preprocessor_path = MODELS_DIR / 'wage_preprocessor.joblib'
    
    if model_path.exists() and preprocessor_path.exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        if metadata_path.exists():
            model_metadata = joblib.load(metadata_path)
        return True
    return False


def predict_wage(state, age, sex, education_level, occupation_code, 
                 years_experience, hours_per_week, class_of_worker):
    """Predict wage based on input characteristics."""
    if model is None or preprocessor is None:
        return None, "Model not loaded"
    
    # Calculate education years and experience
    education_years = EDUCATION_CATEGORY_YEARS.get(education_level, 12)
    
    if years_experience is None or years_experience == '':
        experience = max(0, age - education_years - 6)
    else:
        experience = int(years_experience)
    
    # Default values
    occp_major = occupation_code if occupation_code else '00'
    indp_major = '00'
    
    # Create input dataframe
    input_df = pd.DataFrame([{
        'agep': age,
        'experience': experience,
        'experience_sq': experience ** 2,
        'wkhp': hours_per_week,
        'education_years': education_years,
        'sex': str(sex),
        'education_category': education_level,
        'st': str(state).zfill(2),
        'cow': str(class_of_worker),
        'occp_major': str(occp_major),
        'indp_major': str(indp_major)
    }])
    
    try:
        # Preprocess and predict
        input_processed = preprocessor.transform(input_df)
        log_wage_pred = model.predict(input_processed)[0]
        wage_pred = np.expm1(log_wage_pred)
        return wage_pred, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# Dash App
# ============================================================================

# Initialize app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="PUMS Wage Estimator"
)

# Try to load model on startup
model_loaded = load_model()

# ============================================================================
# Layout
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("💰 PUMS Wage Estimator", className="text-primary mb-2"),
            html.P(
                "Estimate annual wages based on Census Bureau survey data",
                className="lead text-muted"
            ),
        ], className="text-center py-4")
    ]),
    
    # Model Status
    dbc.Row([
        dbc.Col([
            dbc.Alert(
                [
                    html.I(className="bi bi-check-circle-fill me-2"),
                    "Model loaded successfully!" if model_loaded else "⚠️ No model loaded. Please train a model first."
                ],
                color="success" if model_loaded else "warning",
                className="mb-4"
            )
        ])
    ]),
    
    # Input Form
    dbc.Card([
        dbc.CardHeader(html.H4("📝 Enter Your Information", className="mb-0")),
        dbc.CardBody([
            # Row 1: Location, Age, Gender
            dbc.Row([
                dbc.Col([
                    dbc.Label("State"),
                    dcc.Dropdown(
                        id='state-dropdown',
                        options=[{'label': name, 'value': code} 
                                 for code, name in sorted(STATE_CODES.items(), key=lambda x: x[1])],
                        value='06',  # California default
                        placeholder="Select state...",
                        className="mb-3"
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Age"),
                    dbc.Input(
                        id='age-input',
                        type='number',
                        min=18, max=85,
                        value=35,
                        className="mb-3"
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Gender"),
                    dcc.Dropdown(
                        id='sex-dropdown',
                        options=[
                            {'label': 'Male', 'value': 1},
                            {'label': 'Female', 'value': 2}
                        ],
                        value=1,
                        className="mb-3"
                    ),
                ], md=4),
            ]),
            
            # Row 2: Education
            dbc.Row([
                dbc.Col([
                    dbc.Label("Education Level"),
                    dcc.Dropdown(
                        id='education-dropdown',
                        options=[{'label': level, 'value': level} for level in EDUCATION_LEVELS],
                        value="Bachelor's degree",
                        className="mb-3"
                    ),
                ], md=6),
                dbc.Col([
                    dbc.Label("Occupation Category"),
                    dcc.Dropdown(
                        id='occupation-dropdown',
                        options=[{'label': name, 'value': code} 
                                 for code, name in OCCUPATION_CATEGORIES.items()],
                        value='',
                        placeholder="Any occupation...",
                        className="mb-3"
                    ),
                ], md=6),
            ]),
            
            # Row 3: Experience and Hours
            dbc.Row([
                dbc.Col([
                    dbc.Label("Years of Experience"),
                    dbc.Input(
                        id='experience-input',
                        type='number',
                        min=0, max=60,
                        placeholder="Leave blank to estimate from age",
                        className="mb-3"
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Hours per Week"),
                    dbc.Input(
                        id='hours-input',
                        type='number',
                        min=1, max=99,
                        value=40,
                        className="mb-3"
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Class of Worker"),
                    dcc.Dropdown(
                        id='cow-dropdown',
                        options=[{'label': name, 'value': code} 
                                 for code, name in COW_VALUES.items()],
                        value=1,
                        className="mb-3"
                    ),
                ], md=4),
            ]),
            
            # Submit Button
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "🔮 Estimate Wage",
                        id='submit-button',
                        color="primary",
                        size="lg",
                        className="w-100 mt-3"
                    ),
                ], md=6, className="mx-auto")
            ]),
        ])
    ], className="mb-4"),
    
    # Results Section
    html.Div(id='results-container'),
    
    # About Section
    dbc.Card([
        dbc.CardHeader(html.H5("ℹ️ About This Tool", className="mb-0")),
        dbc.CardBody([
            html.P([
                "This wage estimator uses machine learning models trained on the ",
                html.Strong("American Community Survey (ACS) Public Use Microdata Sample (PUMS)"),
                " from the U.S. Census Bureau."
            ]),
            html.P(
                "The model considers factors including location, age, gender, education level, "
                "occupation, and work hours to estimate annual wages."
            ),
            html.P([
                html.Strong("Note: "),
                "Predictions are estimates based on survey data and may not reflect actual wages "
                "for specific individuals or positions."
            ], className="text-muted small mb-0"),
        ])
    ], className="mt-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "Data source: U.S. Census Bureau, American Community Survey PUMS 2023",
                className="text-center text-muted small"
            )
        ])
    ], className="mt-4")
    
], fluid=True, className="py-4")


# ============================================================================
# Callbacks
# ============================================================================

@callback(
    Output('results-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('state-dropdown', 'value'),
    State('age-input', 'value'),
    State('sex-dropdown', 'value'),
    State('education-dropdown', 'value'),
    State('occupation-dropdown', 'value'),
    State('experience-input', 'value'),
    State('hours-input', 'value'),
    State('cow-dropdown', 'value'),
    prevent_initial_call=True
)
def update_prediction(n_clicks, state, age, sex, education, occupation, 
                      experience, hours, cow):
    """Update the wage prediction based on inputs."""
    
    if not all([state, age, sex, education, hours, cow]):
        return dbc.Alert(
            "Please fill in all required fields.",
            color="warning",
            className="mt-4"
        )
    
    # Make prediction
    wage, error = predict_wage(
        state=state,
        age=age,
        sex=sex,
        education_level=education,
        occupation_code=occupation,
        years_experience=experience,
        hours_per_week=hours,
        class_of_worker=cow
    )
    
    if error:
        return dbc.Alert(
            f"Prediction error: {error}",
            color="danger",
            className="mt-4"
        )
    
    if wage is None:
        return dbc.Alert(
            "Could not generate prediction. Please ensure a model is trained.",
            color="warning",
            className="mt-4"
        )
    
    # Calculate derived values
    monthly = wage / 12
    hourly = wage / (hours * 52)
    
    # Confidence range (rough estimate based on typical model MAE)
    mae = 33000  # Approximate MAE from model
    low = max(0, wage - 1.5 * mae)
    high = wage + 1.5 * mae
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H2("Estimated Annual Wage", className="text-center text-muted mb-3"),
                    html.H1(
                        f"${wage:,.0f}",
                        className="text-center text-success display-3 mb-3"
                    ),
                    html.P(
                        f"Range: ${low:,.0f} - ${high:,.0f}",
                        className="text-center text-muted"
                    ),
                ])
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H5("Monthly", className="text-center text-muted"),
                    html.H4(f"${monthly:,.0f}", className="text-center text-primary"),
                ], md=4),
                dbc.Col([
                    html.H5("Hourly (est.)", className="text-center text-muted"),
                    html.H4(f"${hourly:,.2f}", className="text-center text-primary"),
                ], md=4),
                dbc.Col([
                    html.H5("Weekly", className="text-center text-muted"),
                    html.H4(f"${wage/52:,.0f}", className="text-center text-primary"),
                ], md=4),
            ], className="mt-3"),
        ])
    ], color="light", className="mt-4")


# ============================================================================
# Run Server
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PUMS Wage Estimator - Dash Application")
    print("=" * 60)
    print(f"Model loaded: {model_loaded}")
    print(f"Models directory: {MODELS_DIR}")
    print("\nStarting server at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=8050)
