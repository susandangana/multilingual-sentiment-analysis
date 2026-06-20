#======================================================================================================
# PROJECT OVERVIEW
#======================================================================================================
#
# ShopEase is a fast-growing e-commerce company operating across the  united Kingdom, France, Germany,
# and Spain. With a large and growing volume of customer reviews, support requests, and social media 
# feedback, manually analysing customer sentiment has become increasingly challeging.
#
# This project aims to develop a multilingual sentiment analysis solution that automatically processes 
# customer feedback, classifies sentiment as positive, neutral, or negative, and identifies the key 
# drivers of customer satisfation and dissatisfaction. The solution will leverage Natural Language 
# Processing (NLP), machine learning, and transformer-based models to genetate actionable insights that 
# support data-driven decision-making across marketing , product and operations teams.
#
# The project workflow includes data quality assessment, multilingual text preprocessing, exploratory 
# analysis, sentiment modelling, topic extraction, dashboard development, and deployment of scalable 
# sentiemnt monitoring system.
#------------------------------------------------------------------------------------------------------

#======================================================================================================
# %%
# Library Imports
#======================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # used for missing value visualisation
from cleantext import clean
from ftlangdetect import detect
import spacy


#=====================================================================================================
# %%
# Load NLP Models
# #=====================================================================================================
nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')
nlp_de = spacy.load('de_core_news_sm')
nlp_es = spacy.load('es_core_news_sm')

# Create the language model dictionary
language_models = {
    'en': nlp_en,
    'fr': nlp_fr,
    'de': nlp_de,
    'es': nlp_es
}
#=====================================================================================================
# %%
# DATA LOADING AND INITIAL INSPECTION
#=====================================================================================================
df = pd.read_csv('../data/raw_reviews.csv')

# View first five rows of data
print(df.head())

# Check dataset dimension
total_columns = len(df.columns)
total_records = len(df)

print(total_columns)
print(total_records)

# View data type of the columns
print(df.info())

# Validate sentiment labels
unique_sentiments = df['sentiment'].value_counts()
print(unique_sentiments)

total_unique_sentiments = len(unique_sentiments)
print(total_unique_sentiments)

# Check for missing value
print(df.isnull().sum())

# Assign missing review and missing ratings variable names
missing_reviews = df['review'].isnull().sum()
missing_ratings = df['rating'].isnull().sum()

# Check for duplicates
total_duplicates = df.duplicated().sum()
print(total_duplicates)

# Check for invalid ratings
invalid_ratings = df[(df['rating'] < 1) |
                     (df['rating'] > 5)
                     ].shape[0]
print(invalid_ratings)

# Check country inconsistencies
unique_countries = df['country'].unique()
print(unique_countries)

# Number of unique countries
total_unique_countries = len(unique_countries)
print(total_unique_countries)

# Check product category inconsistencies
unique_products = df['product_category'].unique()
print(unique_products)

# Number of unique products
total_unique_products = len(unique_products)
print(total_unique_products)

# Create data dictionaries of original dataset before any preprocessing
# First create the dictionary of the dataset columns and data type
data_dictionary_initial = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes.astype(str)
})

# Add the description to the created data dictionary
data_dictionary_initial['Description'] = [
        'Unique identifier for each review',
        'Customer review text',
        'Customer rating',
        'ISO country code',
        'Product category',
        'Sentiment label',
        'Date and time review was submitted'
    ]

print('data_dictionary_initial')

# Save data dictionary as excel
data_dictionary_initial.to_excel('../outputs/data_dictionary_initial.xlsx',
                                index=False)

# Create Quality Summary Table
quality_summary = pd.DataFrame({
    'Quality Check':[
        'Total Columns',
        'Total Records',
        'Unique Sentiments',
        'Missing Reviews',
        'Missing Ratings',
        'Total Duplicates',
        'Invalid Ratings',
        'Total Unique Countries',
        'Total Unique Products'
    ], 
    'Counts':[
        total_columns,
        total_records,
        total_unique_sentiments,
        total_duplicates,
        missing_reviews,
        missing_ratings,
        invalid_ratings,
        total_unique_countries,
        total_unique_products
    ]
})

print(quality_summary)

# Save quality summary table as excel to file
quality_summary.to_excel('../outputs/quality_summary.xlsx',
                       index=False)
#------------------------------------------------------------------------------------------------------
# Data Quality Summary:
# The dataset contains 12,000 records and 7 variables with no missing reviews, missing ratings, 
# duplicate records, or invalid ratings. Sentiment labels are consistent across three classes, and 
# country values are standardized using ISO country codes. While the project focuses on four core markets, 
# the dataset includes reviews from 14 countries, suggesting a wider international customer base. Overall, 
# minimal data cleaning is required before proceeding to language detection and text preprocessing.
#------------------------------------------------------------------------------------------------------

#======================================================================================================
# %%
# DATA CLEANING 
#======================================================================================================
# Create a copy of the dataset
df_clean = df.copy()

# Convert timestamp from string to datetime format
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

# Check conversion
print(df_clean['timestamp'].dtype)

# View first five records
print(df_clean.head())

# Check review feature thoroughly for any strange charracters
print(df_clean['review'].sample(50, random_state=42))

print(df_clean['review'].head(50))

#  Check the presence of  strange character in row 20 as it exist when datafile is viewed via excel
sample = df_clean.loc[20, 'review']
print(sample)

#------------------------------------------------------------------------------------------------------
# Data Cleaning Summary
# Pottential encoding issues observed in Excel were investigated, but reviews displayed correctly 
# within Python, indicating a file display issue rather than data corruption. No encoding corrections 
# were required. Additionally, the timestamp field was convertaed from string format to datetime format
# to support time-based analysis. The dataset is deemed ready for language detection and text preprocessing.
#------------------------------------------------------------------------------------------------------

#======================================================================================================
# %%
# MULTILINGUAL LANGUAGE DETECTION
#======================================================================================================
# Create a language detection function containing error handling incase the language of a review is unknown

def detect_language(text):

    try:
        result = detect(text)

        return result['lang']
    
    except:
        return 'unknown'


# Tag each review with its language by creating a new column of language in the dataset
df_clean['language'] = df_clean['review'].apply(detect_language)

# Confirm language column creation
print(df_clean[['review', 'language']].sample(10, random_state=42))

# Calculate and view language distribution
language_count = df_clean['language'].value_counts()

# Calculate the percentage distribution of language
language_percentages = ((language_count/len(df_clean))*100).round(2)
print(language_percentages)

# Create Language Summary Table
language_summary = pd.DataFrame({
    'Review Counts':language_count,
    'Percentage':language_percentages
})
print(language_summary)

# Save this summary table as csv in outputs
language_summary.to_excel('../outputs/language_summary.xlsx')

# Validate detection samples by viewing random samples of reviews for each detected language
# View reviews in English
print(df_clean.loc[df_clean['language'] == 'en', 
             'review'].sample(10, random_state=42))

# View reviews in French
print(df_clean.loc[df_clean['language']=='fr', 
                   'review'].sample(10, random_state=42))

# View reviews in German
print(df_clean.loc[df_clean['language']=='de', 
                   'review'].sample(10, random_state=42))

# View reviews in Spanish
print(df_clean.loc[df_clean['language']=='es',
         'review'].sample(10, random_state=42))

# View reviews in Italian
print(df_clean.loc[df_clean['language']=='it',
                   'review'].sample(10, random_state=42))

# View reviews in Portuguese
print(df_clean.loc[df_clean['language']=='pt',
                   'review'].sample(10, random_state=42))
# Reclassify the reviews that are categorised as it and pt as these are actually es
df_clean['language'] = df_clean['language'].replace({
    'it': 'es',
    'pt': 'es'
})

# Confirm this change
print((df_clean['language'].value_counts(normalize=True))*100)

# There seem to be duplicates in the review column. confirm this
print(df_clean['review'].duplicated().sum())

# Further confirmation of duplicate in review column
print(df['review'].nunique())

#============================================================================================================
# Language Detection Insight
# Language detection identified English, French, German, and Spanish as the primary languages in the dataset.
# Manual validation confirmed accurate classification for these major language groups. A small number of reviews
# were initially classified as Italian and Portuguese; however, inspection revealed that these reviews were
# actually Spanish and had likely been misclassified due to linguistic similarities. As these cases represented
# less than 0.05% of the dataset, they were reassigned to Spanish and considered negligible. Additionally,
# although no duplicate records were identified, review text analysis revealed a high degree of repetition,
# suggesting the presence of commonly reused customer feedback phrases. These reviews were retained as they
# represent distinct customer records.
#============================================================================================================

#============================================================================================================
# %%
# PREPROCESSING PIPELINE
#============================================================================================================
# Create a function to perform all of this steps (text cleaning) excluding tokenization
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

# Create function to perform multilingual tokenization, stop-word removal and Lemmaatization
def process_text(row):
    language = row['language']
    text = row['clean_review']

    if language not in language_models:
        return []
    
    doc = language_models[language](text)
    
    return [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

# Apply the above functions to the dataset 
# Create a new feature of the cleaned review- clean_review
df_clean['clean_review'] = df_clean['review'].apply(preprocess_text)
# %%
# Create a new feature of the processed text
# Create a new feature - tokens
df_clean['processed_tokens'] = df_clean.apply(process_text, axis=1)

# Vlidate columns creation by checking random samples of the created columns and review feature
df_clean[['review', 'clean_review', 'processed_tokens']].sample(10, random_state=42)

# View final Dataset Structure
print(df_clean.head())

# Check for missing values in new columns
print(df_clean[['clean_review', 'processed_tokens', 'language']].isnull().sum())

# Save preprocessed dataset to csv
df_clean.to_csv('../data/processed_review.csv')
# %%
# Update data dictionary (adding the created columns alongside their description)
# Create a list of the description of the initial data dictionary
descriptions = data_dictionary_initial['Description'].to_list()

# Update the description with the description of the newly created columns
descriptions.extend([
    'Detected review language',
    'Cleaned review after normalisation',
    'Tokenized, lemmatized text with stop words removed'
])

# Check the length of updated description equals the length of the df_clean columns
print(len(descriptions))
print(len(df_clean.columns))

# Add description to the updated dataframe
data_dictionary_processed = pd.DataFrame({
    'Column Name': df_clean.columns,
    'Data Type': df_clean.dtypes.astype(str),
    'Description': descriptions
})

# View updated data dictionary
print(data_dictionary_processed)

# Save updated data dictionary to excel
data_dictionary_processed.to_excel('data_dictionary_processed',
                                  index=False)
#--------------------------------------------------------------------------------------------------------------
# A multilingual preprocessing pipeline was developed using language-specific spaCy models for English, French, 
# German, and Spanish reviews. The pipeline performed text normalization, tokenization, stop-word removal, and 
# lemmatization while preserving language-specific linguistic rules. Emojis were removed to establish a clean 
# baseline for sentiment analysis and will be reconsidered during model performance comparisons in later 
# stages of the project.
#-------------------------------------------------------------------------------------------------------------
#==============================================================================================================



# %%
