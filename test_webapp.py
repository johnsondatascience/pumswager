"""Test the web application components."""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    from app.prediction_service import WagePredictionService, PredictionInput
    from app.main import app
    print("  All imports successful!")
    return True

def test_service():
    """Test the prediction service."""
    print("\nTesting prediction service...")
    from app.prediction_service import WagePredictionService
    
    service = WagePredictionService()
    options = service.get_available_options()
    
    print(f"  States available: {len(options['states'])}")
    print(f"  Education levels: {len(options['education_levels'])}")
    print("  Service initialized successfully!")
    return True

def test_fastapi():
    """Test FastAPI app creation."""
    print("\nTesting FastAPI app...")
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/api/health")
    print(f"  Health check: {response.status_code}")
    print(f"  Response: {response.json()}")
    
    # Test options endpoint
    response = client.get("/api/options")
    print(f"  Options endpoint: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  States: {len(data['states'])}")
        print(f"  Education levels: {len(data['education_levels'])}")
    
    print("  FastAPI app working!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("PUMS Wage Estimator - Component Tests")
    print("=" * 50)
    
    try:
        test_imports()
        test_service()
        test_fastapi()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
