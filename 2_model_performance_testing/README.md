## SETUP

```
_> cd 2_model_performance_testing
_> poetry install
```

## Model Training

Please find below the relevant information for training the model:

I tried creating a cli command to train the model on local, but could not complete it as I started getting issues on my local system when I attempted local-training. Eventually, I shifted to Google-Colab.

For reference, I am providing the command which can be developed for local training or training using CI:

```
_> poetry run python model_trainer.py --input_path dataset/processed_FAQs.csv --model_name all-MiniLM-L6-v2 --k_fold 5

```

1. We used a SentenceTransformer model(all-MiniLM-L6-v2) in order to train a chatbot with FAQs.
2. We used the cleaned dataset from previous step - [data/processed_FAQs.csv](data/processed_FAQs.csv).
3. The notebook which is used for training the model is located at [jupyter_notebooks/FAQ_Model_Training.ipynb](jupyter_notebooks/FAQ_Model_Training.ipynb).
4. The accuracy, precision and recall of the model is measured in the notebook.
5. We have implemented cross-validation using KFold to reduce over-fitting
6. The trained model is saved at the location [checkpoints/model/trained_chatbot-20240722T154338Z-001](checkpoints/model/trained_chatbot-20240722T154338Z-001)
7. We validated the model outputs with a set of pre-recorded queries and expected responses. This pre-recorded question answer dataset is located at [data/pre_recorded_questions.json](data/pre_recorded_questions.json).
8. We also did some functional testing where we validated certain use-cases:

   - Test response time
   - Test response accuracy
   - Test error management
   - Test context management
   - Test user-friendliness , etc.

### CONCLUSION

The model is OVER-FITTED.
