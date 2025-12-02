"""
Simple Model Training Script for PUMS Wage Prediction

Run this before starting the Dash app to train the prediction model.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Add src to path
sys.path.insert(0, str(BASE_DIR / 'src'))
from codebook import get_education_category

# Education years mapping
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

# Feature configuration
NUMERIC_FEATURES = ['agep', 'experience', 'experience_sq', 'wkhp', 'education_years']
CATEGORICAL_FEATURES = ['sex', 'education_category', 'st', 'cow', 'occp_major', 'indp_major']


def train_model():
    """Train the wage prediction model."""
    print("=" * 60)
    print("PUMS Wage Prediction Model Training")
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
    
    # Feature engineering
    print("\nEngineering features...")
    df['education_years'] = df['schl'].map(EDUCATION_YEARS)
    df['education_category'] = df['schl'].apply(get_education_category)
    df['experience'] = (df['agep'] - df['education_years'] - 6).clip(lower=0)
    df['experience_sq'] = df['experience'] ** 2
    df['log_wagp'] = np.log1p(df['wagp'])
    df['occp_major'] = df['occp'].astype(str).str[:2]
    df['indp_major'] = df['indp'].astype(str).str[:2]
    
    # Convert categorical to string
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    
    # Prepare data
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df_model = df[all_features + ['log_wagp', 'pwgtp']].dropna()
    print(f"Final dataset: {len(df_model):,} records")
    
    X = df_model[all_features]
    y = df_model['log_wagp']
    weights = df_model['pwgtp']
    
    # Split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    print(f"Training: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Preprocessor
    print("\nCreating preprocessor...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed features: {X_train_processed.shape[1]}")
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    print("(This may take a few minutes)")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        verbose=1
    )
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_processed)
    
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"R² Score:     {r2:.4f}")
    print(f"RMSE:         ${rmse:,.0f}")
    print(f"MAE:          ${mae:,.0f}")
    print(f"Sample Size:  {len(df_model):,}")
    
    # Save model
    print("\nSaving model...")
    model_path = MODELS_DIR / 'wage_model_default.joblib'
    preprocessor_path = MODELS_DIR / 'wage_preprocessor_default.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save metadata
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class ModelMetadata:
        model_type: str
        r2_score: float
        rmse: float
        mae: float
        sample_size: int
        training_date: str
    
    metadata = ModelMetadata(
        model_type='gradient_boosting',
        r2_score=r2,
        rmse=rmse,
        mae=mae,
        sample_size=len(df_model),
        training_date=datetime.now().isoformat()
    )
    joblib.dump(metadata, MODELS_DIR / 'wage_metadata_default.joblib')
    
    print(f"\nModel saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the Dash app:")
    print("  python app/dash_app.py")
    
    return True


if __name__ == '__main__':
    train_model()
