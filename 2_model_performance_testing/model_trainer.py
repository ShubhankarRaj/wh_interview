import sys
import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, util, InputExample
# from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

def read_data(path:str):
    df = pd.read_csv(path)
    return df

def train_model(args):

    df = read_data(args.input_path)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    model_name = args.model_name
    model = SentenceTransformer(model_name)

    # Prepare inputExample objects
    examples = []
    for q, a in zip(questions, answers):
        examples.append(InputExample(texts=[q, a]))

    # Define the loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)


    # Define k-fold cross validation
    k = args.k_fold
    kf = KFold(n_splits=k, shuffle=True)
    fold_count = 0

    accuracies = []
    precisions = []
    recalls = []

    for train_index, val_index in kf.split(examples):
        fold_count += 1
        print(f"Fold {fold_count}:")

        train_examples = [examples[i] for i in train_index]
        val_examples = [examples[i] for i in val_index]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        val_dataloader = DataLoader(val_examples, shuffle=True, batch_size=16)

        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Fine-tune the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
            show_progress_bar=True
        )

        # Evaluate the model on validation set
        val_questions = [example.texts[0] for example in val_examples]
        val_answers = [example.texts[1] for example in val_examples]

        val_embeddings = model.encode(val_questions, convert_to_tensor=True)
        question_embeddings = model.encode(questions, convert_to_tensor=True)

        y_true = []
        y_pred = []

        for i, example in enumerate(val_examples):
            cos_scores = util.pytorch_cos_sim(val_embeddings[i], question_embeddings)[0]
            best_match_idx = cos_scores.argmax().item()
            predicted_answer = answers[best_match_idx]
            actual_answer = example.texts[1]
            
            y_true.append(actual_answer)
            y_pred.append(predicted_answer)
            
            print(f"Question: {example.texts[0]}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Actual Answer: {actual_answer}")
            print("-" * 50)

        # Convert answers to binary format for calculating precision and recall
        y_true_binary = [1 if a == p else 0 for a, p in zip(val_answers, y_pred)]
        y_pred_binary = [1 for _ in y_pred]  # All predictions are considered as 1

        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    # Calculate mean accuracy, precision, and recall
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Clean dataset")
    parser.add_argument("--input_path", type=str, help="path to input dataset")
    parser.add_argument("--model_name", type=str, help="Pretrained Sentence Transformer model name")
    parser.add_argument("--k_fold", type=int, help="Enter the count of k folds for Cross validations")
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    
    train_model(args)
