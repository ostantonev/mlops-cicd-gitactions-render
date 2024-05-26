from fastapi.testclient import TestClient
from main import app
from logger_config import log

# Create client test
client = TestClient(app)

def test_init_client():
    """
        Test the server is still running
    """
    response = client.get('/')
    log.info(f"response={response.text}")
    # Test successfull response
    assert response.status_code == 200
