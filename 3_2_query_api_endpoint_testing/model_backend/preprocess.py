import json
import numpy as np
from model_backend.model import model

def load_predefined_questions(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

"""
Preprocesses the questions in the queries and returns the texts and embeddings.

Parameters:
queries (dict): A dictionary containing FAQs with questions.

Returns:
tuple: A tuple containing the preprocessed texts and embeddings.
"""
def preprocess_questions(questions):
    texts = [question['question'] for question in questions['FAQs']]
    embeddings = model.encode(texts)
    return texts, embeddings

def preprocess_single_question(question_text:str):
    text = [question_text]
    embedding = model.encode(text)
    return text, embedding
