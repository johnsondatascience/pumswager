#!/usr/bin/env python
"""
Run the PUMS Wage Estimator web application.

Usage:
    python run_webapp.py [--train] [--host HOST] [--port PORT]

Options:
    --train     Train a new model before starting the server
    --host      Host to bind to (default: 127.0.0.1)
    --port      Port to bind to (default: 8000)
"""

import argparse
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))


def train_model():
    """Train the wage prediction model."""
    from app.prediction_service import WagePredictionService
    
    print("=" * 60)
    print("Training Wage Prediction Model")
    print("=" * 60)
    
    service = WagePredictionService()
    
    try:
        metadata = service.train_model(
            model_type='gradient_boosting',
            use_sample_weights=True
        )
        
        service.save_model('default')
        
        print(f"\nModel trained successfully!")
        print(f"  R² Score: {metadata.r2_score:.4f}")
        print(f"  RMSE: ${metadata.rmse:,.0f}")
        print(f"  MAE: ${metadata.mae:,.0f}")
        print(f"  Sample Size: {metadata.sample_size:,}")
        print(f"\nModel saved to: models/wage_model_default.joblib")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the PUMS data file exists at:")
        print("  data/pums_person_2023.csv")
        print("\nYou can collect data using:")
        print("  python main.py --collect person")
        return False
        
    except Exception as e:
        print(f"\nError training model: {e}")
        return False


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    
    print("=" * 60)
    print("Starting PUMS Wage Estimator Web Application")
    print("=" * 60)
    print(f"\nServer running at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the PUMS Wage Estimator web application"
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model before starting the server'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train the model, do not start the server'
    )
    
    args = parser.parse_args()
    
    # Train model if requested
    if args.train or args.train_only:
        success = train_model()
        if not success and args.train_only:
            sys.exit(1)
        if args.train_only:
            return
    
    # Run the server
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
