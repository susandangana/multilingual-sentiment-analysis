#======================================================================================================
# PROJECT OVERVIEW
#======================================================================================================
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
from cleantext import clean
from ftlangdetect import detect
import spacy

from data_preprocessing_utils import process_text_for_prediction

#=====================================================================================================
# %%
# RAW DATA LOADING AND INITIAL INSPECTION
#=====================================================================================================
df = pd.read_csv('../data/unprocessed_reviews.csv')

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

# %%
# Number of unique sentiments
total_unique_sentiments = len(unique_sentiments)
print(total_unique_sentiments)

# %%
# Check for missing value
print(df.isnull().sum())
missing_records = df.isnull().sum().sum()

# %%
# Check for duplicates
total_duplicates = df.duplicated().sum()
print(total_duplicates)

# %%
# Check for invalid ratings
invalid_ratings = df[(df['rating'] < 1) |
                     (df['rating'] > 5)
                     ].shape[0]
print(invalid_ratings)

# %%
# Check country inconsistencies and number of countries
unique_countries = df['country'].unique()
print(unique_countries)

# Number of unique countries
total_unique_countries = df['country'].nunique()
print(total_unique_countries)

# %%
# Check product category inconsistencies and number of products
unique_products = df['product_category'].unique()
print(unique_products)

# Number of unique products
total_unique_products = len(unique_products)
print(total_unique_products)

# %%
# Create data dictionaries of original dataset before any preprocessing
# First create the dictionary of the dataset columns and data type
data_dictionary_initial = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes.astype(str)
})

# Add the description to the created data dictionary
data_dictionary_initial['Description'] = [
        'Unique identifier for each review',
        'Categories of products being reviewed',
        'Date and time review was submitted',
        'ISO country code',
        'Customer rating',
        'Customer review text',
        'Sentiment label',
        
    ]

print(data_dictionary_initial)

# Save data dictionary as excel
data_dictionary_initial.to_excel('../outputs/data_dictionary_initial.xlsx',
                                index=False)

# %%
# Create Quality Summary Table
quality_summary = pd.DataFrame({
    'Quality Check':[
        'Total Columns',
        'Total Records',
        'Unique Sentiments',
        'Missing Records',
        'Total Duplicates',
        'Invalid Ratings',
        'Total Unique Countries',
        'Total Unique Products'
    ], 
    'Counts':[
        total_columns,
        total_records,
        total_unique_sentiments,
        missing_records,
        total_duplicates,
        invalid_ratings,
        total_unique_countries,
        total_unique_products
    ]
})

print(quality_summary)

# Save quality summary table as excel to file
quality_summary.to_excel('../outputs/quality_summary.xlsx',
                       index=False)


#======================================================================================================
# %%
# DATA CLEANING 
#======================================================================================================
# Create a copy of the dataset
df_clean = df.copy()
# %%
# Convert timestamp from string to datetime format
df_clean['timestamp'] = pd.to_datetime(
    df_clean['timestamp'], format='ISO8601')

# Check conversion
print(df_clean['timestamp'].dtype)
print(df_clean['timestamp'].head())


# Convert it to a standard datetime and remove the timezone
df_clean['timestamp'] = (df_clean['timestamp']
                         .dt.tz_localize(None)
                         )
print(df_clean['timestamp'].head())

# %%
# Check review feature thoroughly for any strange charracters
print(df_clean['review'].sample(50, random_state=42))

print(df_clean['review'].head(50))

#  Check the presence of  strange character in row 20 as it exist when datafile is viewed via excel
sample = df_clean.loc[20, 'review']
print(sample)


#======================================================================================================
# %%
# MULTILINGUAL LANGUAGE DETECTION and RECLASSIFICATION
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


# %%
# View initial language distribution
language_count = df_clean['language'].value_counts()
print(language_count)

# %%
# Calculate the percentage distribution of language
language_percentages = ((language_count/len(df_clean))*100).round(3)
print(language_percentages)

# %%
# Manual validation of  detected languages - by viewing random samples of reviews for each detected language
# Create function for viewing samples of reviews in detected language
def view_review_samples(df, lang, col1, col2, num):

    available = len(df[df[col1]==lang])

    sample = df.loc[df[col1]==lang,
                    col2].sample(min(num, available), random_state=42)
    return sample

# Create a list of the detected languages
langs = df_clean['language'].unique().tolist()

# Call the function to view review samples
for lang in langs:
    print(f"\nReviews in {lang}")
    sample = view_review_samples(df_clean, lang, 'language', 'review', 10)
    print(sample)


# Lowercase transformation - check effect of review casing on the language detection
# %%
# Convert review to lower case and then detect language again
df_clean['review_lower'] = df_clean['review'].str.lower()

# Confirm new feature creation with lower case texts
print(df_clean.head())

# %%
# Perform language detection again
df_clean['language_update'] = df_clean['review_lower'].apply(detect_language)

# %%
# View language_update distribution and compare with initial language distribution 
language_count_update = df_clean['language_update'].value_counts()
# Get the percentage of the updated language distribution
language_percent = df_clean['language_update'].value_counts(normalize=True)*100

# Combine the before and after language distribution
comparison = pd.concat(
    [language_count, language_percentages, language_count_update, language_percent],
    axis=1
)

comparison.columns = ['Before','Before%', 'After', 'After%']
comparison = comparison.fillna(0)
# Make the before and after integer, and the percentages rounded
comparison[['Before', 'After']] = comparison[['Before', 'After']].astype(int)
comparison[['Before%', 'After%']] = (
    comparison[['Before%', 'After%']].round(3)
    )

print(comparison)

# %%
# Create Language Summary Table
language_summary = pd.DataFrame({
    'Review Counts':language_count_update,
    'Percentage':language_percent
})
print(language_summary)

# Save this summary table as csv in outputs
language_summary.to_excel('../outputs/language_summary.xlsx')

# %%
# Before vs After normalization language classification visualisation
# focus on the other languages that changed (since the english language dominates the other language)
focus_langs = ['ja', 'de', 'es', 'pl', 'zh', 'ml', 'ru']

# Use the above languages to plot a barchart
comparison.loc[focus_langs][['Before', 'After']].plot(
    kind='bar',
    figsize=(10, 5)
)

plt.title('Language Distribution Before and After Normalisation Excluding English (lowercasing)')
plt.xlabel('Language')
plt.ylabel('Review Count')
plt.xticks(rotation=0)
plt.legend(title='')
plt.tight_layout()
plt.show()


# Multilingual Reclassification and Removal Decision

# %% 
# Create a list of the updated detected languages
langs = df_clean['language_update'].unique().tolist()

# Call the function to view review samples
for lang in langs:
    print(f"\nReviews in {lang}")
    sample_lower = view_review_samples(df_clean, lang, 'language_update', 'review_lower', 21)
    print(sample_lower)

# about 75% of the Polish (pl) detected reviews are English. 
# %%
# Reclassify to en, the three english review of the four reviews classified as pl
# Create list of these
pl_en = [14337, 16987, 20815]

df_clean.loc[
    df_clean.index.isin(pl_en),
      'language_update'
      ] = 'en'

# Reclassify the reviews that are categorised as it and pt as these are actually es
df_clean['language_update'] = df_clean['language_update'].replace({
    'ja': 'en',
    'ru': 'en',

})

# Remove the sv and cs as these are gibberish
df_clean = df_clean[
    ~df_clean['language_update'].isin(['sv', 'cs'])
]

# Confirm reclassifications
# Get language distribution
language_update = df_clean['language_update'].value_counts()
language_percent_update = df_clean['language_update'].value_counts(normalize=True)*100

# Create Updated Language Summary Table after reclassifications
language_summary_update = pd.DataFrame({
    'Review Counts':language_update,
    'Percentage':language_percent_update
})
print(language_summary_update)

# Save this summary table as csv in outputs
language_summary_update.to_excel('../outputs/language_summary_update.xlsx')


# %%
# Create a new feature of the processed text using the process function from
# data_preprocessing_utils script
df_clean['processed_texts'] = (
    df_clean['review']
    .fillna('')
    .apply(process_text_for_prediction)
)
# %%
# Validate columns creation by checking random samples of the created columns and review feature
df_clean[['review', 'processed_texts']].sample(10, random_state=42)

# View final Dataset Structure
print(df_clean.head())

# Confirm there are no missing values in new columns
print(df_clean[['processed_texts', 'language_update']].isnull().sum())
# %%
# Save preprocessed dataset to csv
df_clean.to_csv('../data/processed_reviews.csv',
                index=False)

# %%
# Update data dictionary (adding the created columns alongside their description)
# Create a list of the description of the initial data dictionary
descriptions = data_dictionary_initial['Description'].to_list()


# Update the description with the description of the newly created columns
descriptions.extend([
    'Detected review language',
    'Normalized review',
    'updated review language after reclassification',
    'Joined tokenized, lemmatized text with stop words removed'
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
data_dictionary_processed.to_excel('../outputs/data_dictionary_processed.xlsx')

#--------------------------------------------------------------------------------------------------------------
# Data Quality Summary
#-----------------------
# The dataset contains 21,055 records and 7 variables with one missing record in the country feature only.
# Sentiment labels are consistent across three classes, and country values are standardized using ISO 
# country codes. The dataset includes reviews from 148 countries, suggesting a wider international 
# customer base. Overall, minimal data cleaning is required before proceeding to language detection 
# and text preprocessing.

# Data Cleaning Summary
#-----------------------
# Pottential encoding issues observed in Excel were investigated, but reviews displayed correctly 
# within Python, indicating a file display issue rather than data corruption. No encoding corrections 
# were required. Additionally, the timestamp field was convertaed from string format to datetime format
# to support time-based analysis. The dataset is deemed ready for language detection and text preprocessing.

# Multilanguae Detection
#------------------------
# Initial language detection identified 12 language groups, with English accounting for 99.64% of reviews.
# Manual validation revealed several language detection errors caused by uppercase English text and noisy 
# reviews.
# Reviews were converted to lowercase and language detection was rerun, increasing English reviews from 20,979
# to 21,006 and reducing Japanese classifications from 26 to 1.
# Further validation identified English reviews incorrectly classified as Japanese, Russian and Polish, which
# were reclassified to English. Gibberish Swedish and Czech records were removed.
# Final dataset contains 21,011 English reviews (99.80%) and 42 genuine multilingual reviews across 7 language 
# groups

# Multilanguage preprocessing 
#-----------------------------
# A multilingual preprocessing pipeline was developed using language-specific spaCy models for English, French, 
# German, and Spanish reviews. The pipeline performed text normalization, tokenization, stop-word removal, and 
# lemmatization while preserving language-specific linguistic rules. Emojis were removed to establish a clean 
# baseline for sentiment analysis and will be reconsidered during model performance comparisons in later 
# stages of the project.
#-------------------------------------------------------------------------------------------------------------
