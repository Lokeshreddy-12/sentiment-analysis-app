import pytest
from app import app


def test_api_sentiment_success():
    client = app.test_client()
    response = client.post('/api/sentiment', json={'text': 'I love this product!'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'sentiment' in data
    assert 'score' in data
    assert 'model_type' in data


def test_api_sentiment_missing_text():
    client = app.test_client()
    response = client.post('/api/sentiment', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data