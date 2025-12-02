"""Train the wage prediction model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Starting model training...")
print("=" * 60)

from app.prediction_service import WagePredictionService

service = WagePredictionService()

print(f"Data directory: {service.data_dir}")
print(f"Models directory: {service.models_dir}")

# Check if data exists
data_file = service.data_dir / 'pums_person_2023.csv'
print(f"Data file exists: {data_file.exists()}")

if not data_file.exists():
    print("ERROR: Data file not found!")
    sys.exit(1)

print("\nLoading and training model...")
print("This may take a few minutes...")

try:
    metadata = service.train_model(
        model_type='gradient_boosting',
        use_sample_weights=True
    )
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"R² Score: {metadata.r2_score:.4f}")
    print(f"RMSE: ${metadata.rmse:,.0f}")
    print(f"MAE: ${metadata.mae:,.0f}")
    print(f"Sample Size: {metadata.sample_size:,}")
    
    # Save model
    model_path = service.save_model('default')
    print(f"\nModel saved to: {model_path}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
