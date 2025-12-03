"""Test the web application components."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    
    assert len(options['states']) > 0, "No states available"
    assert len(options['education_levels']) > 0, "No education levels available"
    
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
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    print(f"  Health check: {response.status_code}")
    print(f"  Response: {response.json()}")
    
    # Test options endpoint
    response = client.get("/api/options")
    assert response.status_code == 200, f"Options endpoint failed: {response.status_code}"
    print(f"  Options endpoint: {response.status_code}")
    
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
