"""Check setup and train model."""
import sys
import os

# Force output to be unbuffered
sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

print("Step 1: Checking imports...")

try:
    import pandas as pd
    print("  pandas OK")
except ImportError as e:
    print(f"  pandas FAILED: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("  numpy OK")
except ImportError as e:
    print(f"  numpy FAILED: {e}")
    sys.exit(1)

try:
    from sklearn.ensemble import GradientBoostingRegressor
    print("  scikit-learn OK")
except ImportError as e:
    print(f"  scikit-learn FAILED: {e}")
    sys.exit(1)

try:
    import joblib
    print("  joblib OK")
except ImportError as e:
    print(f"  joblib FAILED: {e}")
    sys.exit(1)

print("\nStep 2: Checking data file...")
from pathlib import Path
data_path = Path(__file__).parent / 'data' / 'pums_person_2023.csv'
print(f"  Path: {data_path}")
print(f"  Exists: {data_path.exists()}")

if data_path.exists():
    # Get file size
    size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

print("\nStep 3: Testing prediction service import...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.prediction_service import WagePredictionService
    print("  Import OK")
except Exception as e:
    print(f"  Import FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Creating service instance...")
try:
    service = WagePredictionService()
    print("  Service created OK")
    print(f"  Data dir: {service.data_dir}")
    print(f"  Models dir: {service.models_dir}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\nStep 5: Loading sample data...")
try:
    df = pd.read_csv(data_path, nrows=1000)
    print(f"  Loaded {len(df)} sample rows")
    print(f"  Columns: {list(df.columns)[:10]}...")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All checks passed! Ready to train model.")
print("=" * 50)
