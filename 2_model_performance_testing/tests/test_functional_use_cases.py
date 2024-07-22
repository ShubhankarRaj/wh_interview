from app.model import model
from app.model_response import get_model_response
import pytest

@pytest.fixture
def get_model():
    return model

def test_response_time():
    import time
    start_time = time.time()
    response = get_model_response(question_text="How do I reset my device?")
    end_time = time.time()
    assert (end_time - start_time) < 5  # Ensure response time is less than 5 seconds

def test_response_accuracy():
    response = get_model_response(question_text="How do I reset my device?")
    assert "reset" in response.lower()  # Check if the response contains relevant information

def test_error_management():
    response = get_model_response(question_text="askjdhasjkdh")
    assert "I'm sorry" in response or "I don't understand" in response  # Ensure graceful handling of errors

def test_conversational_flow():
    _ = get_model_response(question_text="Hello")
    response2 = get_model_response(question_text="I have an issue with my screen")
    assert "screen" in response2.lower()  # Ensure the bot maintains context

def test_user_friendliness():
    response = get_model_response(question_text="Tell me about your warranty services")
    assert "warranty" in response.lower()  # Ensure the response is clear and relevant

def test_understanding():
    response = get_model_response(question_text="My phone won't turn on")
    assert "turn on" in response.lower()  # Ensure the bot understands the problem

# def test_multi_platform_compatibility():
#     response_web = get_model_response(question_text="How do I connect to WiFi?", platform="web")
#     response_mobile = get_model_response(question_text="How do I connect to WiFi?", platform="mobile")
#     assert response_web == response_mobile  # Ensure consistent responses across platforms

def test_input_validation():
    response = get_model_response(question_text="Can I get a warranty for 2 years?")
    assert "warranty" in response.lower() and ("yes" in response.lower() or "no" in response.lower())  # Validate response

def test_graphic_content_display():
    response = get_model_response(question_text="Show me the wiring diagram")
    assert "diagram" in response.lower()  # Check for proper handling of graphical content

def test_query_length_handling():
    short_query = get_model_response(question_text="Help")
    long_query = get_model_response(question_text="I need help with my device because it is not charging properly and I have tried multiple chargers")
    assert len(short_query) > 0 and len(long_query) > 0  # Ensure bot handles both short and long queries

# def test_graceful_failure(get_model):
#     get_model.simulate_failure()  # Simulate a failure scenario
#     response = get_model.get_response("Why isn't it working?")
#     assert "error" in response.lower() or "try again" in response.lower()  # Check for graceful failure handling

# def test_crash_prevention(get_model):
#     for _ in range(1000):
#         get_model.get_response("Test crash prevention")
#     assert get_model.is_running()  # Ensure the bot is still running after stress testing

def test_parameter_input_integrity(get_model):
    # response = get_model_response(question_text="Can you help me John?", user_data={"name": "John"})
    response = get_model_response(question_text="Can you help me John?")
    assert "John" in response  # Ensure user data is correctly handled

def test_thorough_testing(get_model):
    response = get_model_response(question_text="Tell me about your data handling policies")
    assert "data" in response.lower() and "secure" in response.lower()  # Check for proper data handling information

if __name__ == "__main__":
    pytest.main()