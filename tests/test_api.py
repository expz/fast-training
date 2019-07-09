import warnings
import falcon.testing
import pytest
from app import api


@pytest.fixture
def client():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    return falcon.testing.TestClient(api)


def test_translate_api(client: falcon.testing.TestClient):
    fr_text = 'Comment allez-vous?'
    response = client.simulate_post('/api/translate/fren', json={'fr': fr_text})
    assert response.status_code == 200
    assert isinstance(response.json['en'], str)
    assert 'How are you?' in response.json['en']
