from app.model import model
from sentence_transformers import util
from app.preprocess import load_predefined_questions

def get_model_responses(predefined_questions_filepath):
    questions = load_predefined_questions(predefined_questions_filepath)
    test_question_texts = [question['question'] for question in questions['FAQs']]
    
    test_question_embeddings = model.encode(test_question_texts)
    all_question_embeddings = model.encode(model.get_train_question_answers()[0])
    
    y_pred = []
    y_actual = []

    for i, question in enumerate(questions['FAQs']):
        cos_scores = util.pytorch_cos_sim(test_question_embeddings[i], all_question_embeddings)[0]
        best_match_idx = cos_scores.argmax().item()
        predicted_answer = model.get_train_question_answers()[1][best_match_idx]
        actual_answer = question['answer']
        y_actual.append(actual_answer)
        y_pred.append(predicted_answer)
        # print(f"Question: {question['question']}")
        # print(f"Predicted Answer: {predicted_answer}")
        # print(f"Actual Answer: {actual_answer}")
        
    return y_actual, y_pred


def get_model_response(question_text):
    test_question_text = [question_text]
    test_question_embedding = model.encode(test_question_text)
    all_question_embeddings = model.encode(model.get_train_question_answers()[0])

    cos_scores = util.pytorch_cos_sim(test_question_embedding, all_question_embeddings)[0]
    best_match_idx = cos_scores.argmax().item()
    predicted_response = model.get_train_question_answers()[1][best_match_idx]
    
    return predicted_response
