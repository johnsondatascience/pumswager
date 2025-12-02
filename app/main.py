"""
Wage Prediction API

FastAPI backend for the wage estimation web service.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

from prediction_service import (
    WagePredictionService,
    PredictionInput,
    get_prediction_service,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PUMS Wage Estimator",
    description="Estimate annual wages based on demographic and employment characteristics using Census PUMS data",
    version="1.0.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class WagePredictionRequest(BaseModel):
    """Request model for wage prediction."""
    state: str = Field(..., description="State FIPS code (e.g., '06' for California)")
    age: int = Field(..., ge=18, le=85, description="Age in years")
    sex: int = Field(..., ge=1, le=2, description="Sex (1=Male, 2=Female)")
    education_level: str = Field(..., description="Education level category")
    field_of_degree: Optional[str] = Field(None, description="Field of degree code")
    occupation_code: Optional[str] = Field(None, description="2-digit occupation code")
    years_experience: Optional[int] = Field(None, ge=0, le=60, description="Years of experience")
    hours_per_week: int = Field(40, ge=1, le=99, description="Hours worked per week")
    class_of_worker: int = Field(1, ge=1, le=9, description="Class of worker code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "state": "06",
                "age": 35,
                "sex": 1,
                "education_level": "Bachelor's degree",
                "occupation_code": "15",
                "hours_per_week": 40,
                "class_of_worker": 1
            }
        }


class WagePredictionResponse(BaseModel):
    """Response model for wage prediction."""
    predicted_wage: float = Field(..., description="Predicted annual wage in dollars")
    confidence_interval_low: float = Field(..., description="Lower bound of confidence interval")
    confidence_interval_high: float = Field(..., description="Upper bound of confidence interval")
    model_r2: float = Field(..., description="Model R² score")
    sample_size: int = Field(..., description="Number of records used to train model")
    formatted_wage: str = Field(..., description="Formatted wage string")
    formatted_range: str = Field(..., description="Formatted confidence interval")


class TrainModelRequest(BaseModel):
    """Request model for training a new model."""
    model_name: str = Field("default", description="Name for the trained model")
    model_type: str = Field("gradient_boosting", description="Model type")
    states: Optional[List[str]] = Field(None, description="Filter to specific states")
    min_age: Optional[int] = Field(None, description="Minimum age filter")
    max_age: Optional[int] = Field(None, description="Maximum age filter")
    min_sample_weight: Optional[float] = Field(None, description="Minimum survey weight")
    use_sample_weights: bool = Field(True, description="Use survey weights in training")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_type: str
    r2_score: float
    rmse: float
    mae: float
    sample_size: int
    training_date: str
    is_loaded: bool


class OptionsResponse(BaseModel):
    """Response model for available input options."""
    states: Dict[str, str]
    sex: Dict[int, str]
    education_levels: List[str]
    class_of_worker: Dict[int, str]


# API Endpoints

@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "PUMS Wage Estimator API", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    service = get_prediction_service()
    return {
        "status": "healthy",
        "model_loaded": service.model is not None
    }


@app.get("/api/options", response_model=OptionsResponse)
async def get_options():
    """Get available options for input fields."""
    service = get_prediction_service()
    options = service.get_available_options()
    return OptionsResponse(
        states=options['states'],
        sex={int(k): v for k, v in options['sex'].items()},
        education_levels=options['education_levels'],
        class_of_worker={int(k): v for k, v in options['class_of_worker'].items()}
    )


@app.get("/api/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the currently loaded model."""
    service = get_prediction_service()
    
    if service.metadata is None:
        return ModelInfoResponse(
            model_type="none",
            r2_score=0.0,
            rmse=0.0,
            mae=0.0,
            sample_size=0,
            training_date="N/A",
            is_loaded=False
        )
    
    return ModelInfoResponse(
        model_type=service.metadata.model_type,
        r2_score=service.metadata.r2_score,
        rmse=service.metadata.rmse,
        mae=service.metadata.mae,
        sample_size=service.metadata.sample_size,
        training_date=service.metadata.training_date,
        is_loaded=service.model is not None
    )


@app.post("/api/predict", response_model=WagePredictionResponse)
async def predict_wage(request: WagePredictionRequest):
    """
    Predict annual wage based on input characteristics.
    
    Returns predicted wage with confidence interval.
    """
    service = get_prediction_service()
    
    if service.model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Please train a model first using /api/model/train"
        )
    
    try:
        prediction_input = PredictionInput(
            state=request.state,
            age=request.age,
            sex=request.sex,
            education_level=request.education_level,
            field_of_degree=request.field_of_degree,
            occupation_code=request.occupation_code,
            years_experience=request.years_experience,
            hours_per_week=request.hours_per_week,
            class_of_worker=request.class_of_worker
        )
        
        result = service.predict(prediction_input)
        
        return WagePredictionResponse(
            predicted_wage=result.predicted_wage,
            confidence_interval_low=result.confidence_interval_low,
            confidence_interval_high=result.confidence_interval_high,
            model_r2=result.model_r2,
            sample_size=result.sample_size,
            formatted_wage=f"${result.predicted_wage:,.0f}",
            formatted_range=f"${result.confidence_interval_low:,.0f} - ${result.confidence_interval_high:,.0f}"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/train")
async def train_model(request: TrainModelRequest):
    """
    Train a new wage prediction model.
    
    Optionally filter data by state, age range, or survey weight.
    """
    service = get_prediction_service()
    
    # Build subset criteria
    subset_criteria = {}
    if request.states:
        subset_criteria['states'] = request.states
    if request.min_age:
        subset_criteria['min_age'] = request.min_age
    if request.max_age:
        subset_criteria['max_age'] = request.max_age
    if request.min_sample_weight:
        subset_criteria['min_sample_weight'] = request.min_sample_weight
    
    try:
        metadata = service.train_model(
            subset_criteria=subset_criteria if subset_criteria else None,
            model_type=request.model_type,
            use_sample_weights=request.use_sample_weights
        )
        
        # Save the model
        service.save_model(request.model_name)
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "r2_score": metadata.r2_score,
            "rmse": metadata.rmse,
            "mae": metadata.mae,
            "sample_size": metadata.sample_size,
            "message": f"Model trained successfully with {metadata.sample_size:,} records"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/load")
async def load_model(name: str = Query("default", description="Model name to load")):
    """Load a previously trained model."""
    service = get_prediction_service()
    
    if service.load_model(name):
        return {
            "status": "success",
            "message": f"Model '{name}' loaded successfully"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found"
        )


# Mount static files for frontend
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
