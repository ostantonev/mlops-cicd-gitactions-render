from fastapi.testclient import TestClient
from main import app
from logger_config import log

# Create client test
client = TestClient(app)

def test_alive_client():
    """
        Test the server is alive
    """
    response = client.get('/')
    log.info(f"response={response.text}")
    # Test successfull response
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/json'
    assert response.json() == {"Hello": "World(main)"}
