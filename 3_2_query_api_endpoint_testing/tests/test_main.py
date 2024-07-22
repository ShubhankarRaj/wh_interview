import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from model_service.main import app

client = TestClient(app)

def test_query_endpoint():
    response = client.post("/query", json={"query": "What is the capital of France?"})
    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], str)

def test_query_endpoint_invalid_input():
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], str)

def test_query_endpoint_missing_input():
    response = client.post("/query", json={})
    assert response.status_code == 422 
