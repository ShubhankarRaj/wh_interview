from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

class ModelWrapper:
    def __init__(self, model_name: str=None, model_path: str=None, data_path: str=None):
        # Untrained Model
        if model_name:
            self.model = SentenceTransformer(model_name)
        else:
            # Trained Model
            self.model = SentenceTransformer(model_path)
        self.data_path = data_path
    
    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)

    def find_most_similar(self, query_embedding, candidates_embeddings):
        similarities = np.inner(query_embedding, candidates_embeddings)
        return np.argmax(similarities)
    
    def get_train_question_answers(self):
        df = pd.read_csv(self.data_path)
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        return questions, answers


model_path = 'model_checkpoint/trained_chatbot-20240722T154338Z-001'
# Initialize the model
# model = ModelWrapper(model_name='all-MiniLM-L6-v2')
model = ModelWrapper(model_path=model_path, data_path='data/processed_FAQs.csv')
