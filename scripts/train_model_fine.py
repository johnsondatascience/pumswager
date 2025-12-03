"""
Fine-Grained Model Training Script for PUMS Wage Prediction

Trains a LightGBM model with a more detailed field-of-degree feature
and saves it as a separate "fine" model alongside the default model.

Usage:
    python scripts/train_model_fine.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import sys

# Paths - handle running from scripts/ or project root
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Add src to path
sys.path.insert(0, str(BASE_DIR / 'src'))
from codebook import get_education_category, FIELD_OF_DEGREE_VALUES

# Education years mapping (same as default script)
EDUCATION_YEARS = {
    1: 0, 2: 0, 3: 0,
    4: 1, 5: 2, 6: 3, 7: 4,
    8: 5, 9: 6, 10: 7, 11: 8,
    12: 9, 13: 10, 14: 11, 15: 11,
    16: 12, 17: 12,
    18: 13, 19: 14,
    20: 14,
    21: 16,
    22: 18,
    23: 20,
    24: 21
}

# Base feature configuration (from Phase 1)
NUMERIC_FEATURES = ['agep', 'experience', 'experience_sq', 'wkhp', 'education_years']
BASE_CATEGORICAL_FEATURES = ['sex', 'education_category', 'st', 'cow', 'occp_major', 'indp_major']

# Fine-grained additional categorical features
FINE_CATEGORICAL_ADDITIONAL = ['field_detailed']
CATEGORICAL_FEATURES_FINE = BASE_CATEGORICAL_FEATURES + FINE_CATEGORICAL_ADDITIONAL


def map_field_detailed(code) -> str:
    """Map FOD1P code to a detailed field-of-degree label.

    Uses FIELD_OF_DEGREE_VALUES; missing/NA gets a stable fallback.
    """
    if pd.isna(code):
        return "No degree / N/A"
    # Codes are 4-digit; ensure zero-padded string
    try:
        code_str = str(int(code)).zfill(4)
    except (ValueError, TypeError):
        code_str = str(code).zfill(4)
    return FIELD_OF_DEGREE_VALUES.get(code_str, "Other / Unknown")


def train_model_fine():
    """Train the fine-grained wage prediction model."""
    print("=" * 60)
    print("PUMS Wage Prediction Model Training (FINE)")
    print("=" * 60)
    
    # Load data
    data_path = DATA_DIR / 'pums_person_2023.csv'
    print(f"\nLoading data from: {data_path}")
    
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Please run: python main.py --collect person")
        return False
    
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} records")
    
    # Filter to working population
    print("\nFiltering to working population...")
    df = df[
        (df['wagp'] > 0) &
        (df['agep'] >= 18) & (df['agep'] <= 70) &
        (df['esr'].isin([1, 2, 4, 5]))
    ].copy()
    print(f"Filtered to {len(df):,} workers")
    
    # Feature engineering (shared with default model)
    print("\nEngineering features (base + fine-grained)...")
    df['education_years'] = df['schl'].map(EDUCATION_YEARS)
    df['education_category'] = df['schl'].apply(get_education_category)
    df['experience'] = (df['agep'] - df['education_years'] - 6).clip(lower=0)
    df['experience_sq'] = df['experience'] ** 2
    df['log_wagp'] = np.log1p(df['wagp'])
    df['occp_major'] = df['occp'].astype(str).str[:2]
    df['indp_major'] = df['indp'].astype(str).str[:2]
    
    # Fine-grained field-of-degree feature
    # FOD1P is the primary field-of-degree code in PUMS
    df['field_detailed'] = df['fod1p'].apply(map_field_detailed)
    
    # Convert categorical to string
    for col in CATEGORICAL_FEATURES_FINE:
        df[col] = df[col].astype(str)
    
    # Prepare data
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES_FINE
    df_model = df[all_features + ['log_wagp', 'pwgtp']].dropna()
    print(f"Final dataset (fine): {len(df_model):,} records")
    
    X = df_model[all_features]
    y = df_model['log_wagp']
    weights = df_model['pwgtp']
    
    # Split (preserve survey weights for train/test)
    print("\nSplitting data (fine model)...")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    print(f"Training: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Preprocessor
    print("\nCreating preprocessor (fine model)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES_FINE)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed features (fine): {X_train_processed.shape[1]}")
    
    # Train model
    print("\nTraining LightGBM fine-grained model...")
    print("(This may take a few minutes)")
    
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    # Fit with survey weights
    model.fit(X_train_processed, y_train, sample_weight=w_train)
    
    # Evaluate with survey weights
    print("\nEvaluating fine model (survey-weighted)...")
    y_pred = model.predict(X_test_processed)
    
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)
    
    r2 = r2_score(y_test, y_pred, sample_weight=w_test)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual, sample_weight=w_test))
    mae = mean_absolute_error(y_test_actual, y_pred_actual, sample_weight=w_test)
    
    print("\n" + "=" * 60)
    print("FINE MODEL PERFORMANCE")
    print("=" * 60)
    print(f"R² Score:     {r2:.4f}")
    print(f"RMSE:         ${rmse:,.0f}")
    print(f"MAE:          ${mae:,.0f}")
    print(f"Sample Size:  {len(df_model):,}")
    
    # Save fine model
    print("\nSaving fine model...")
    model_path = MODELS_DIR / 'wage_model_fine.joblib'
    preprocessor_path = MODELS_DIR / 'wage_preprocessor_fine.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save metadata as a simple dict (pickle-safe)
    from datetime import datetime

    metadata = {
        'model_type': 'lightgbm_fine',
        'r2_score': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'sample_size': int(len(df_model)),
        'training_date': datetime.now().isoformat(),
        'feature_names': all_features,
    }
    joblib.dump(metadata, MODELS_DIR / 'wage_metadata_fine.joblib')
    
    print(f"\nFine model saved to: {model_path}")
    print(f"Fine preprocessor saved to: {preprocessor_path}")
    print("\n" + "=" * 60)
    print("FINE MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can load this model in the API with:")
    print("  POST /api/model/load?name=fine")
    
    return True


if __name__ == '__main__':
    train_model_fine()
