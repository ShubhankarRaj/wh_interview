import pytest
from app.model_response import get_model_responses

def test_model_accuracy():
    y_actual, y_pred = get_model_responses('data/pre_recorded_questions.json')
    total = len(y_actual)

    correct = 0
    for i in range(total):
        if y_actual[i] == y_pred[i]:
            correct += 1

    accuracy = correct / total
    assert accuracy > 0.8