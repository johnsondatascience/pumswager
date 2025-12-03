"""
Wage Prediction Service

Handles model loading, training, and predictions for the wage estimation web service.
Supports training models on data subsets based on sample size and survey weights.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from codebook import (
    STATE_CODES,
    EDUCATION_CATEGORIES,
    SEX_VALUES,
    COW_VALUES,
    FIELD_OF_DEGREE_VALUES,
    get_education_category,
)

logger = logging.getLogger(__name__)

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

# Reverse mapping for education category to years
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


@dataclass
class PredictionInput:
    """Input data for wage prediction."""
    state: str  # State code (e.g., '06' for California)
    age: int
    sex: int  # 1=Male, 2=Female
    education_level: str  # Education category string
    field_of_degree: Optional[str] = None  # Field of degree code
    occupation_code: Optional[str] = None  # 2-digit occupation major group
    years_experience: Optional[int] = None  # If not provided, estimated from age/education
    hours_per_week: int = 40
    class_of_worker: int = 1  # Default: private for-profit


@dataclass
class PredictionResult:
    """Result of wage prediction."""
    predicted_wage: float
    confidence_interval_low: float
    confidence_interval_high: float
    model_r2: float
    sample_size: int
    features_used: List[str]


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""
    model_type: str
    r2_score: float
    rmse: float
    mae: float
    sample_size: int
    training_date: str
    feature_names: List[str]
    subset_criteria: Optional[Dict[str, Any]] = None


class WagePredictionService:
    """Service for wage prediction using PUMS data."""
    
    def __init__(self, models_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path(__file__).parent.parent / 'models'
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data'
        self.models_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.preprocessor = None
        self.metadata: Optional[ModelMetadata] = None
        
        # Feature configuration
        self.numeric_features = [
            'agep', 'experience', 'experience_sq', 'wkhp', 'education_years'
        ]
        self.categorical_features = [
            'sex', 'education_category', 'st', 'cow', 'occp_major', 'indp_major'
        ]
        
    def load_data(self, subset_criteria: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Load PUMS person data with optional filtering.
        
        Args:
            subset_criteria: Optional dict with filtering criteria like:
                - states: List of state codes to include
                - min_age, max_age: Age range
                - education_levels: List of education categories
                - min_sample_weight: Minimum survey weight threshold
        """
        data_path = self.data_dir / 'pums_person_2023.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path, low_memory=False)
        logger.info(f"Loaded {len(df):,} records from {data_path}")
        
        # Base filtering: working population with positive wages
        df = df[
            (df['wagp'] > 0) &
            (df['agep'] >= 18) & (df['agep'] <= 70) &
            (df['esr'].isin([1, 2, 4, 5]))
        ].copy()
        
        # Apply subset criteria
        if subset_criteria:
            if 'states' in subset_criteria:
                df = df[df['st'].astype(str).str.zfill(2).isin(subset_criteria['states'])]
            if 'min_age' in subset_criteria:
                df = df[df['agep'] >= subset_criteria['min_age']]
            if 'max_age' in subset_criteria:
                df = df[df['agep'] <= subset_criteria['max_age']]
            if 'min_sample_weight' in subset_criteria:
                df = df[df['pwgtp'] >= subset_criteria['min_sample_weight']]
        
        logger.info(f"After filtering: {len(df):,} records")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling."""
        df = df.copy()
        
        # Create derived features
        df['education_years'] = df['schl'].map(EDUCATION_YEARS)
        df['education_category'] = df['schl'].apply(get_education_category)
        df['experience'] = (df['agep'] - df['education_years'] - 6).clip(lower=0)
        df['experience_sq'] = df['experience'] ** 2
        df['log_wagp'] = np.log1p(df['wagp'])
        
        # Occupation and industry major groups
        df['occp_major'] = df['occp'].astype(str).str[:2]
        df['indp_major'] = df['indp'].astype(str).str[:2]
        
        # Ensure string types for categorical
        for col in self.categorical_features:
            df[col] = df[col].astype(str)
        
        return df
    
    def train_model(
        self,
        subset_criteria: Optional[Dict[str, Any]] = None,
        model_type: str = 'gradient_boosting',
        min_records: int = 1000,
        use_sample_weights: bool = True
    ) -> ModelMetadata:
        """
        Train a wage prediction model.
        
        Args:
            subset_criteria: Optional filtering criteria for data subset
            model_type: 'gradient_boosting', 'random_forest', 'ridge', or 'lightgbm'
            min_records: Minimum number of records required for training
            use_sample_weights: Whether to use survey weights in training
            
        Returns:
            ModelMetadata with training results
        """
        # Load and prepare data
        df = self.load_data(subset_criteria)
        
        if len(df) < min_records:
            raise ValueError(
                f"Insufficient data: {len(df)} records (minimum: {min_records}). "
                "Consider broadening subset criteria."
            )
        
        df = self.prepare_features(df)
        
        # Check weight distribution
        total_weight = df['pwgtp'].sum()
        logger.info(f"Total survey weight: {total_weight:,.0f}")
        
        # Prepare features and target
        all_features = self.numeric_features + self.categorical_features
        df_model = df[all_features + ['log_wagp', 'wagp', 'pwgtp']].dropna()
        
        X = df_model[all_features]
        y = df_model['log_wagp']
        weights = df_model['pwgtp'] if use_sample_weights else None
        
        # Train/test split
        if weights is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            w_train, w_test = None, None
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 self.categorical_features)
            ]
        )
        
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Select and train model
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            )
        elif model_type == 'lightgbm':
            self.model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train with or without sample weights
        if w_train is not None and hasattr(self.model, 'fit'):
            try:
                self.model.fit(X_train_processed, y_train, sample_weight=w_train)
            except TypeError:
                # Model doesn't support sample_weight
                self.model.fit(X_train_processed, y_train)
        else:
            self.model.fit(X_train_processed, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_processed)
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(y_pred)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        
        # Get feature names
        cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(
            self.categorical_features
        )
        all_feature_names = list(self.numeric_features) + list(cat_feature_names)
        
        # Create metadata
        from datetime import datetime
        self.metadata = ModelMetadata(
            model_type=model_type,
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            sample_size=len(df_model),
            training_date=datetime.now().isoformat(),
            feature_names=all_feature_names,
            subset_criteria=subset_criteria
        )
        
        logger.info(f"Model trained: R²={r2:.4f}, RMSE=${rmse:,.0f}, MAE=${mae:,.0f}")
        
        return self.metadata
    
    def save_model(self, name: str = 'default') -> Path:
        """Save the trained model and preprocessor."""
        if self.model is None or self.preprocessor is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_path = self.models_dir / f'wage_model_{name}.joblib'
        preprocessor_path = self.models_dir / f'wage_preprocessor_{name}.joblib'
        metadata_path = self.models_dir / f'wage_metadata_{name}.joblib'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        joblib.dump(self.metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, name: str = 'default') -> bool:
        """Load a trained model and preprocessor."""
        model_path = self.models_dir / f'wage_model_{name}.joblib'
        preprocessor_path = self.models_dir / f'wage_preprocessor_{name}.joblib'
        metadata_path = self.models_dir / f'wage_metadata_{name}.joblib'
        
        if not model_path.exists():
            # Try legacy paths
            model_path = self.models_dir / 'wage_prediction_model.joblib'
            preprocessor_path = self.models_dir / 'wage_preprocessor.joblib'
            if not model_path.exists():
                logger.warning(f"No model found at {model_path}")
                return False
        
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        if metadata_path.exists():
            loaded_meta = joblib.load(metadata_path)
            # Support both dict-based metadata (from scripts/train_model.py)
            # and ModelMetadata instances
            if isinstance(loaded_meta, dict):
                self.metadata = ModelMetadata(
                    model_type=loaded_meta.get('model_type', 'unknown'),
                    r2_score=float(loaded_meta.get('r2_score', 0.0)),
                    rmse=float(loaded_meta.get('rmse', 0.0)),
                    mae=float(loaded_meta.get('mae', 0.0)),
                    sample_size=int(loaded_meta.get('sample_size', 0)),
                    training_date=str(loaded_meta.get('training_date', 'unknown')),
                    feature_names=list(loaded_meta.get('feature_names', [])),
                    subset_criteria=loaded_meta.get('subset_criteria'),
                )
            else:
                self.metadata = loaded_meta
        else:
            # Create minimal metadata for legacy models
            self.metadata = ModelMetadata(
                model_type='unknown',
                r2_score=0.0,
                rmse=0.0,
                mae=0.0,
                sample_size=0,
                training_date='unknown',
                feature_names=[],
            )
        
        logger.info(f"Model loaded from {model_path}")
        return True
    
    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """
        Predict wage for given input characteristics.
        
        Args:
            input_data: PredictionInput with user characteristics
            
        Returns:
            PredictionResult with predicted wage and confidence interval
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        # Calculate experience if not provided
        education_years = EDUCATION_CATEGORY_YEARS.get(input_data.education_level, 12)
        if input_data.years_experience is not None:
            experience = input_data.years_experience
        else:
            experience = max(0, input_data.age - education_years - 6)
        
        # Default occupation/industry codes if not provided
        occp_major = input_data.occupation_code or '00'
        indp_major = '00'  # Default industry
        
        # Create input dataframe
        input_df = pd.DataFrame([{
            'agep': input_data.age,
            'experience': experience,
            'experience_sq': experience ** 2,
            'wkhp': input_data.hours_per_week,
            'education_years': education_years,
            'sex': str(input_data.sex),
            'education_category': input_data.education_level,
            'st': str(input_data.state).zfill(2),
            'cow': str(input_data.class_of_worker),
            'occp_major': str(occp_major),
            'indp_major': str(indp_major)
        }])
        
        # Preprocess and predict
        input_processed = self.preprocessor.transform(input_df)
        log_wage_pred = self.model.predict(input_processed)[0]
        wage_pred = np.expm1(log_wage_pred)
        
        # Estimate confidence interval (rough approximation using model MAE)
        mae = getattr(self.metadata, 'mae', 30000) if self.metadata else 30000
        ci_low = max(0, wage_pred - 1.5 * mae)
        ci_high = wage_pred + 1.5 * mae
        
        return PredictionResult(
            predicted_wage=float(wage_pred),
            confidence_interval_low=float(ci_low),
            confidence_interval_high=float(ci_high),
            model_r2=getattr(self.metadata, 'r2_score', 0.0) if self.metadata else 0.0,
            sample_size=getattr(self.metadata, 'sample_size', 0) if self.metadata else 0,
            features_used=self.numeric_features + self.categorical_features
        )
    
    def get_available_options(self) -> Dict[str, Any]:
        """Get available options for input fields."""
        return {
            'states': {code: name for code, name in STATE_CODES.items()},
            'sex': SEX_VALUES,
            'education_levels': list(EDUCATION_CATEGORY_YEARS.keys()),
            'class_of_worker': COW_VALUES,
            'fields_of_degree': FIELD_OF_DEGREE_VALUES,
        }


# Singleton instance for the web service
_service_instance: Optional[WagePredictionService] = None


def get_prediction_service() -> WagePredictionService:
    """Get or create the prediction service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = WagePredictionService()
        # Try to load existing model
        _service_instance.load_model()
    return _service_instance
