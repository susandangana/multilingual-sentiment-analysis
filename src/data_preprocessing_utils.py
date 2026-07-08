#======================================================================================================
# %%
# Library Imports
#======================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cleantext import clean
from ftlangdetect import detect
import spacy


#============================================================================================================
# %%
# PREPROCESSING PIPELINE
#============================================================================================================

# Load NLP Models

nlp_en = spacy.load('en_core_web_sm')
nlp_it = spacy.load('it_core_news_sm')


# Create the language model dictionary
language_models = {
    'en': nlp_en,
    'it': nlp_it,
}
# Create a function to perform text cleaning
def preprocess_text(text):
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False, 
        lower=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=True,
        no_emoji=True
    )

    return text


# %%
# Create function to perform multilingual tokenization, Stop word removal and Lematization
def process_text_for_prediction(text):

    clean_review = preprocess_text(text)
    language = detect(clean_review)["lang"]

    doc = get_nlp_model(language)(clean_review)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    return ' '.join(tokens)
# %%

# Create helper function to use language-specific model if available, otherwise fall back to English model
def get_nlp_model(language):
    return language_models.get(language, nlp_en)

