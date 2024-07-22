from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from user_service.main import app, get_db
import user_service.user_model as models
import user_service.user_schema as user_schema
mock_session = MagicMock()


@pytest.fixture
def db_mock():
    mock_session = MagicMock()
    mock_session.query.reset_mock()

    return mock_session

@pytest.fixture
def override_get_db(db_mock):
    app.dependency_overrides[get_db] = lambda: db_mock
    yield
    app.dependency_overrides[get_db] = get_db


client = TestClient(app)

def test_create_user(db_mock):

    # Mock the behavior of `get_user_by_email` to return None (indicating no existing user)
    db_mock.query.return_value.filter.return_value.first.return_value = None

    # Mock the `create_user` function to simulate successful creation
    with patch('user_service.crud.get_user_by_email', return_value=None) as mock_get_user_by_email:
        with patch('user_service.crud.create_user') as mock_create_user:
            mock_create_user.return_value = models.User(id=1, email="johndoe@example.com", name="John Doe", gender="Male", age=24)
            
            with patch('user_service.main.get_db', return_value=db_mock):
                response = client.post("/users/", json={"email": "johndoe@example.com", "name": "John Doe", "age": 24, "gender": "Male"})
            
            data = response.json()
            print(data)
            assert response.status_code == 200
            
            assert data["id"] == 1
            assert data["name"] == "John Doe"
            assert data["email"] == "johndoe@example.com"
            assert data["age"] == 24

def test_get_user(db_mock):
    with patch('user_service.crud.get_user', return_value=models.User(id=1, email="johndoe@example.com", name="John Doe", gender="Male", age=24)) as mock_get_user_by_email:
        with patch('user_service.main.get_db', return_value=db_mock):
            response = client.get("/users/1")
        data = response.json()
        assert response.status_code == 200
        assert data["id"] == 1
        assert data["name"] == "John Doe"
        assert data["email"] == "johndoe@example.com"
        assert data["age"] == 24

def test_create_existing_user(db_mock):
    db_mock.query.return_value.filter.return_value.first.return_value = models.User(id=1, email="johndoe@example.com", name="John Doe", gender="Male", age=24)

    with patch('user_service.crud.get_user_by_email', return_value=models.User(id=1, email="johndoe@example.com", name="John Doe", gender="Male", age=24)) as mock_get_user_by_email:
        with patch('user_service.crud.create_user') as mock_create_user:
            mock_create_user.return_value = models.User(id=1, email="johndoe@example.com", name="John Doe", gender="Male", age=24)
            
            with patch('user_service.main.get_db', return_value=db_mock):
                response = client.post("/users/", json={"email": "johndoe@example.com", "name": "John Doe", "age": 24, "gender": "Male"})
            
            data = response.json()
            assert response.status_code == 400
            assert data["detail"] == "Email already registered"


def test_query_users_by_name(override_get_db, db_mock):
    # Arrange
    mock_user = user_schema.User(id=1, name="John Doe", email="john@example.com", gender="male", age=30)
    db_mock.query.return_value.filter.return_value.all.return_value = [mock_user]

    # Act
    response = client.post("/query_users/", json={"email": "john@example.com"})

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "John Doe"
    assert data[0]["email"] == "john@example.com"


def test_query_users_by_gender(override_get_db, db_mock):
    # Arrange
    mock_user = user_schema.User(id=1, name="John Doe", email="john@example.com", gender="male", age=30)
    db_mock.query.return_value.filter.return_value.all.return_value = [mock_user]

    # Act
    response = client.post("/query_users/", json={"gender": "male"})

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "John Doe"
    assert data[0]["email"] == "john@example.com"

