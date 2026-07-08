#======================================================================================================
# %%
# Library Imports
#======================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report)
from sklearn.preprocessing import LabelEncoder
import torch
import transformers
import datasets
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

#===========================================================================================================
# %%
# BASELINE MODEL DEVELOPMENT 
#===========================================================================================================
# %%
# Load preprocessed dataset
df_clean = pd.read_csv('../data/processed_reviews.csv')

# View dataset
print(df_clean.head())

# %%
# Check for missing value in the newly created column and remove if any
print(df_clean['processed_texts'].isna().sum())

# Drop these missing values as they are blanks read as missing value when csv was read
df_clean = df_clean.dropna(subset=['processed_texts'])

# Confirm drop
print(df_clean['processed_texts'].isna().sum())

# %%
# Data preparation
# Seperate the feature (processed_text) from the target (sentiment)
X = df_clean['processed_texts']
y = df_clean['sentiment']


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000
)

# %%
# Fit vectorizer to train feature, and transform both train and text features
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# %%
# Naive Bayes Model Training
#----------------------------

# Initialize model
nb_model = MultinomialNB()

# Fit model to vectorized train set
nb_model.fit(
    X_train_tfidf,
    y_train
)

# Make prediction on vectorized test set
nb_pred = nb_model.predict(
    X_test_tfidf
)

# Evaluate model performance
# View accuracy score
print(
    accuracy_score(
        y_test,
        nb_pred
    )
)

# View classification report
print(
    classification_report(
        y_test,
        nb_pred
    )
)

# Save trained Naive Bayes model
joblib.dump(
    nb_model,
    '../models/naive_bayes.pkl'
)

# %%
# Logistic Regression Model Training
#------------------------------------

# Initialize logistic regression
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Fit model
lr_model.fit(
    X_train_tfidf,
    y_train
    )

# Make prediction with model
lr_pred = lr_model.predict(
    X_test_tfidf
)

# Evaluate model performance
# View model accuracy
print(
    accuracy_score(
        y_test,
        lr_pred
    )
)

# View classification report
print(
    classification_report(
        y_test,
        lr_pred
    )
)
# %%
# Save Logistic regression model without weight balanced
joblib.dump(
    lr_model,
    '../models/logistic_regression.pkl'
)
# %%
# Save the TF-IDF Vectorizer
joblib.dump(
    tfidf,
    '../models/tfidf_vectorizer.pkl'
)

# %%
# Calibrate the logistic weight and re-run
lr_model2 = LogisticRegression(
    class_weight = 'balanced',
    max_iter=1000,
    random_state=42
)

# Fit model to train set
lr_model2.fit(
    X_train_tfidf,
    y_train
)

# Make predection
lr_pred2 = lr_model2.predict(
    X_test_tfidf
)

# Evaluate model performance
print(
    accuracy_score(
        y_test,
        lr_pred2
    )
)

print(
    classification_report(
        y_test,
        lr_pred2
    )
)

# %%
# Visual Comparison
baseline_results = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression'],
    'Accuracy': [0.86, 0.89],
    'F1 Score': [0.84, 0.87]

})

baseline_results.plot(
    x='Model',
    y=['Accuracy', 'F1 Score'],
    kind='bar',
    figsize=(6,4)
)

plt.title('Baseline Model Performance')
plt.ylabel('Score')
plt.ylim(0.75, 1.0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Transformer model developer
#------------------------------

# Inspect the reveiw and clean_review features for contraction such as won't, don't etc.
print(df_clean[['review', 'clean_review']].sample(10))

# prepare dataset again this time using the 'review' feature instead of the 'processed_texts'
# Seperate feature from target
X = df_clean['review']
y = df_clean['sentiment']

# Split datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Encode the target (sentiment)
le = LabelEncoder()

y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# %%
# Confirm encoding
print(dict(zip(le.classes_,
           le.transform(le.classes_))))

# %%
# DistilBERT Pipeline

# Create Hugging Face Datasets
df_train = pd.DataFrame({
    'text': X_train,
    'label': y_train_enc
})

df_test = pd.DataFrame({
    'text': X_test,
    'label': y_test_enc
})

dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)

print(dataset_train)
print(dataset_test)

# %%
# Remove the pandas dataframe index from the created dataset
dataset_train = dataset_train.remove_columns(
    '__index_level_0__'
)

dataset_test = dataset_test.remove_columns(
    '__index_level_0__'
)

# Confirm removal
print(dataset_train)
print(dataset_test)

# %%
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased'
)

# Test tokenizer
sample = tokenizer(
    dataset_train[0]['text'],
    truncation=True,
    padding='max_length', 
    max_length=128
)

print(sample.keys())
# inspect actual review
print(dataset_train[0]['text'])

# %%
# Tokenize entire Dataset
# Create function to tokenize dataset
def tokenize_review(data):
    return tokenizer(
        data['text'],
        truncation=True,
        padding='max_length',
        max_length=128
        )

# Apply function to entire train and test datasets
tokenized_train = dataset_train.map(
    tokenize_review,
    batched=True
)

tokenized_test = dataset_test.map(
    tokenize_review,
    batched=True
)


# Verify tokenization
print(tokenized_train)
print(tokenized_test)

# %%
# Remove the raw text column from dataset
tokenized_train = tokenized_train.remove_columns(
    ['text']
)

tokenized_test = tokenized_test.remove_columns(
    ['text']
)

# Confirm removeal
print(tokenized_train)
print(tokenized_test)

# %%
# Convert dataset to Pytorch format
tokenized_train.set_format('torch')
tokenized_test.set_format('torch')

# Verify conversion
print(tokenized_train[0])
print(tokenized_test[0])

# %%
# Load DistilBERT
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)

# Define Evaluation Metrics
accuracy = evaluate.load('accuracy')
precision = evaluate.load('precision')
recall = evaluate.load('recall')
f1 = evaluate.load('f1')

# Create a function to comput evaluation metrics
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    return {
        'accuracy': accuracy.compute(
            predictions=predictions,
            references=labels
        )['accuracy'],

        'precision': precision.compute(
            predictions=predictions,
            references=labels,
            average='weighted'
        )['precision'],

        'recall': recall.compute(
            predictions=predictions,
            references=labels,
            average='weighted'
        )['recall'],

        'f1': f1.compute(
            predictions=predictions,
            references=labels,
            average='weighted'
        )['f1']
    }

# %%
# Load traininig arguments
training_args = TrainingArguments(
    output_dir='../models/distilbert_results',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir='../models/logs',
    logging_steps=100
)

# %%
# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# %%
# Run evaluation
trainer.evaluate()

# %% 
# Generate predictions
predictions = trainer.predict(tokenized_test)

y_pred = np.argmax(
    predictions.predictions,
    axis=1
)

# View classification report
print(
    classification_report(
        y_test_enc,
        y_pred,
        target_names=le.classes_
    )
)


# %%
# Start from a fresh model and run for epoch=2
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# Increase epoch from 1 to 2 in training arguments and retrain transformer
training_args = TrainingArguments(
    output_dir='../models/distilbert_results_epoch2',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir='../models/logs_epoch2',
    logging_steps=100
)

# Recreate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)


# Retrain
trainer.train()

# Run evaluation
trainer.evaluate()

# Generate predictions
predictions = trainer.predict(tokenized_test)

y_pred_e2 = np.argmax(
    predictions.predictions,
    axis=1
)


# View classification report
print(
    classification_report(
        y_test_enc,
        y_pred_e2,
        target_names=le.classes_    
    )
)


# Save model and tokenizer
trainer.save_model('..models/final_distilbert')
tokenizer.save_pretrained('../models/final_distilbert')

# %%
# Model comparison (Logistic Regression and DistiBert)
model_results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'DistilBERT\n(2 Epochs)'
    ],
    'Accuracy': [0.89, 0.92],
    'F1 Score': [0.87,  0.91]
})

# Plot
ax = model_results.plot(
    x='Model',
    y=['Accuracy', 'F1 Score'],
    kind='bar',
    figsize=(8, 5)
)

plt.title('Model Comparison - Logistic Regression vs DistilBERT')
plt.xticks(rotation=0)
plt.show()

#----------------------------------------------------------------------------------------------------------
# Base Model Performance
#------------------------
# Logistic Regression outperformed Naïve Bayes across all overall evaluation metrics.
# Both models classified Negative reviews effectively, achieving F1-scores above 0.90.
# Logistic Regression improved Positive review classification, increasing the F1-score from 0.79 to 0.84.
# Both models struggled to identify Neutral reviews due to the severe class imbalance in the dataset.
# To address this, Logistic Regression was retrained using class_weight='balanced', which improved Neutral
# recall from 0.01 to 0.37 but reduced overall accuracy from 88.7% to 81.4%. Given the project's objective 
# of identifying dissatisfied customers, the standard Logistic Regression model was selected as the final 
# model due to its superior Negative review detection and strongest overall overall performance.

# Transformer Model Performace relative to the Best performing base model
#-------------------------------------------------------------------------
# Although the original dataset contained multiple languages, language validation and cleaning resulted in 
# 99.8% of reviews being classified as English. Consequently, an English pre-trained DistilBERT model was 
# selected as it is better aligned with the linguistic characteristics of the final dataset while also 
# providing computational efficiency.
#
# DistilBERT achieved the strongest overall performance among all evaluated models, reaching 91% accuracy 
# and a weighted F1-score of 0.90. The model classified Negative and Positive reviews effectively, achieving 
# F1-scores of 0.95 and 0.89 respectively.
#
# Although DistilBERT improved Neutral class detection compared to the standard Logistic Regression model 
# (F1-score increased from 0.01 to 0.10), performance remained limited due to the severe class imbalance in 
# the dataset. Overall, DistilBERT was selected as the final model because it delivered the best balance of 
# accuracy, precision, recall and F1-score across the sentiment classes.
#----------------------------------------------------------------------------------------------------------

# %%
