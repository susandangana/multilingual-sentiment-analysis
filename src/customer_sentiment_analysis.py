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
import missingno as msno # used for missing value visualisation
from cleantext import clean
from ftlangdetect import detect
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

#=====================================================================================================
# %%
# DATA LOADING AND INITIAL INSPECTION
#=====================================================================================================
df = pd.read_csv('../data/amazon_reviews_cleaned.csv')

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
#------------------------------------------------------------------------------------------------------
# Data Quality Summary:
# The dataset contains 21,055 records and 7 variables with one missing record in the country feature only.
# Sentiment labels are consistent across three classes, and country values are standardized using ISO 
# country codes. The dataset includes reviews from 148 countries, suggesting a wider international 
# customer base. Overall, minimal data cleaning is required before proceeding to language detection 
# and text preprocessing.
#------------------------------------------------------------------------------------------------------

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

# %%
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
# Before vs After language classification visualisation
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


# %%
# Multilingual Reclassification and Removal Decision
#======================================

# %% 
# Call the function to view samples of the review again
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

#============================================================================================================
# Initial language detection identified 12 language groups, with English accounting for 99.64% of reviews.
#
# Manual validation revealed several language detection errors caused by uppercase English text and noisy 
# reviews.
#
# Reviews were converted to lowercase and language detection was rerun, increasing English reviews from 20,979
# to 21,006 and reducing Japanese classifications from 26 to 1.
#
# Further validation identified English reviews incorrectly classified as Japanese, Russian and Polish, which
# were reclassified to English. Gibberish Swedish and Czech records were removed.
#
# Final dataset contains 21,011 English reviews (99.80%) and 42 genuine multilingual reviews across 7 language 
# groups
#============================================================================================================

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

# Create helper function to use language-specific model if available, otherwise fall back to English model
def get_nlp_model(language):
    return language_models.get(language, nlp_en)


# Create function to perform multilingual tokenization, stop-word removal and Lemmaatization
def process_text(row):
    language = row['language_update']
    text = row['clean_review']
    
    
    doc = get_nlp_model(language)(text)
    
    return [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

# Apply the above functions to the dataset 
# Create a new feature of the cleaned review- clean_review
df_clean['clean_review'] = df_clean['review_lower'].apply(
    preprocess_text
    )

# Create a new feature of the processed text
# Create a new feature - tokens
df_clean['processed_tokens'] = df_clean.apply(
    process_text, 
    axis=1
    )

# %%
# Vlidate columns creation by checking random samples of the created columns and review feature
df_clean[['review', 'clean_review', 'processed_tokens']].sample(10, random_state=42)

# View final Dataset Structure
print(df_clean.head())

# Check for missing values in new columns
print(df_clean[['clean_review', 'processed_tokens', 'language_update']].isnull().sum())

# Save preprocessed dataset to csv
df_clean.to_csv('../data/processed_review.csv')

# %%
# Update data dictionary (adding the created columns alongside their description)
# Create a list of the description of the initial data dictionary
descriptions = data_dictionary_initial['Description'].to_list()


# Update the description with the description of the newly created columns
descriptions.extend([
    'Detected review language',
    'Normalized review',
    'updated review language after reclassification',
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
data_dictionary_processed.to_excel('../outputs/data_dictionary_processed.xlsx')
#--------------------------------------------------------------------------------------------------------------
# A multilingual preprocessing pipeline was developed using language-specific spaCy models for English, French, 
# German, and Spanish reviews. The pipeline performed text normalization, tokenization, stop-word removal, and 
# lemmatization while preserving language-specific linguistic rules. Emojis were removed to establish a clean 
# baseline for sentiment analysis and will be reconsidered during model performance comparisons in later 
# stages of the project.
#-------------------------------------------------------------------------------------------------------------

#==============================================================================================================
# EXPLORATORY DATA ANALYSIS
#==============================================================================================================
# %%
# Univariate Analysis 
#---------------------

# Sentiment Distribution (Sentiment Balance Analysis)
review_sentiment_dist = df_clean['sentiment'].value_counts()

print(df_clean['sentiment'].value_counts())

# Visualise review distribution across sentiment classes
ax = review_sentiment_dist.plot(
    kind='bar',
    figsize=(10, 5)
    )

# Total number of reviews
total_review = review_sentiment_dist.sum()

# Add percentage label
for p in ax.patches:
    percentage = (p.get_height()/total_review)*100

    ax.annotate(
        f'{percentage:.1f}%',
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center',
        va='bottom'
    )

plt.title('Review Distribution Across Sentiment Classes')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

#-------------------------------------------------------------------------------------------------------------
# Sentiment distribution is heavily skewed towards negative reviews, with 68.2% of reviews classified as 
# negative. Positive reviews account for 27.6%, while neutral reviews represent only 4.2% of the dataset, 
# indicating a strong imbalance in customer sentiment.
#-------------------------------------------------------------------------------------------------------------

# Bivariate Analysis
#---------------------
# %%
# Sentiment Distribution by Product Category
product_sentiment_dist_pct = (pd.crosstab(
    df_clean['product_category'],
    df_clean['sentiment'],
    normalize='index'
)*100).round(2)

print(product_sentiment_dist_pct)

# Visualise sentiment distribution by product category
ax = product_sentiment_dist_pct.plot(
    kind='bar',
    stacked=True,
    figsize=(10,6)
)

plt.title('Sentiment Distribution by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------------------
# Home & Living recorded the highest negative sentiment (70.0%).
# Fashion recorded the highest positive sentiment (29.4%).
# Negative sentiment exceeded 66% in every category.
# The variation across categories is relatively small, suggesting customer dissatisfaction is a platform-wide 
# issue rather than being concentrated in a specific product category.
#-------------------------------------------------------------------------------------------------------------

# %%
# Check review volume by country i.e Review distribution by country
country_dist = df_clean['country'].value_counts()

# View the top 20 country by review volume
print(country_dist.head(20))

# %%
# Get the records of the top 10 countries by volume
top_countries = df_clean['country'].value_counts().head(10).index

# extract the records of the top 10 country by review volume
df_top = df_clean[df_clean['country'].isin(top_countries)
                  ]
print(df_top)

# Calculate the proportion of the total review that is from the top 10 countries
top10_totalreview_pct = 

# %%
# Create Country-Sentiment Table
country_sentiment_dist_pct = (pd.crosstab(
    df_top['country'],
    df_top['sentiment'],
    normalize='index'
)*100).round(2)

print(country_sentiment_dist_pct)

# Visualize distribution
country_sentiment_dist_pct.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 5)
)

plt.title('Sentiment Distribution by Country (Top 10)')
plt.xlabel('Country')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show
#-----------------------------------------------------------------------------------------------------------
# There are a total of 19128 review records for the top 10 countries by volume
# the top 10 countries are US, GB, CA, IN, IE, DK, NL, AU, DE, IT. 
# Total reviews from this country makes up 
# US and GB account for 78.8% of all reviews
# Canada has the highest negative sentiment (77.7%)
# Italy has the highest positive sentiment (56.0%)
# Germany also shows relatively balanced sentiment (45.9% positive)
# Most major markets exhibit predominantly negative sentiment
#-----------------------------------------------------------------------------------------------------------

# %%
# Get top Positive and Negative keywords from cleaned review (tokenized review)
# Convert tokenized list back to string and create a new feature of this strings
df_clean['processed_texts'] = df_clean['processed_tokens'].apply(
    lambda x: ' '.join(x)
)

# Confirm new feature creation
print(df_clean['processed_texts'].head())

# %%
# Extract top negative words
# Get the processed_texts column of the negative reviews
negative_reviews = df_clean.loc[
    df_clean['sentiment']=='Negative', 
    'processed_texts'
    ]

# Confirm creation of series of negative_reviews 
print(negative_reviews.head(10))

# %%
# Identify top 20 negative keywords from the negative texts column
# Create the TF-IDF vectorizer and configure to keep only the top 20 most important words
tfidf = TfidfVectorizer(
    max_features=20
)

X = tfidf.fit_transform(negative_reviews)

negative_keywords = pd.DataFrame({
    'Keyword':tfidf.get_feature_names_out(),
    'Score':X.sum(axis=0).A1
}).sort_values(
    'Score',
    ascending=False
)

# View dataframe of the top 20 negative key words
print(negative_keywords)

# %%
# Extract top positive words


# %%
