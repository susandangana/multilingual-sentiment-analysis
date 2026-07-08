#======================================================================================================
# %%
# Library Imports
#======================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

from sklearn.feature_extraction.text import TfidfVectorizer

# %%
# Ensure process reviews file path exists
print(os.path.exists('../data/processed_reviews.csv'))

#==============================================================================================================
# EXPLORATORY DATA ANALYSIS
#==============================================================================================================
# %%
# Load preprocessed dataset
df_clean = pd.read_csv('../data/processed_reviews.csv')

# View loaded dataset
print(df_clean.head())

# %%
# Univariate Analysis 
#---------------------

# Sentiment Distribution (Sentiment Balance Analysis)
sentiment_dist = df_clean['sentiment'].value_counts()

print(df_clean['sentiment'].value_counts())

# Visualise review distribution across sentiment classes
ax = sentiment_dist.plot(
    kind='bar',
    figsize=(10, 5)
    )

# Total number of reviews
total_review = sentiment_dist.sum()

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


# %%
# Bivariate Analysis
#---------------------
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


# %%
# Check review volume by country i.e Review distribution by country
country_dist = df_clean['country'].value_counts()

# View the top 20 country by review volume
print(country_dist.head(20))

# %%
# Get the records of the top 10 countries by volume
top_countries = df_clean['country'].value_counts().head(10).index

# extract the records of the top 10 country by review volume
df_top10 = df_clean[df_clean['country'].isin(top_countries)
                  ]
print(df_top10)

# %%
# Calculate the proportion of the total review that is from the top 10 countrie
top10_pct = round(
    len(df_top10)/len(df_clean)*100,
    2
)
print(top10_pct)

# %%
# Create Country-Sentiment Table
country_sentiment_dist_pct = (pd.crosstab(
    df_top10['country'],
    df_top10['sentiment'],
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

# %%
# Ngram Analysis
#----------------
# Convert tokenized list back to string and create a new feature of this strings

# First we ensure processed_tokens columns which became a string instead of list of tokens when saved
# to csv is treated properly as list of tokens when applying the join method

df_clean['processed_tokens'] = (
    df_clean['processed_tokens']
    .apply(ast.literal_eval)
)

df_clean['processed_texts'] = df_clean['processed_tokens'].apply(
    lambda x: ' '.join(x)
)

# Confirm new feature creation
print(df_clean['processed_texts'].head())

# Save the updated clean data
df_clean.to_csv('../data/processed_reviews.csv',
                index=False)

# Check for missing value in the newly created column and remove if any
df_clean['processed_texts'].isna().sum()

# %%
# Create a reusable ngram function
def get_ngram_terms(
        text, 
        ngram=(1,1), 
        max_features=20,
        stop_words=None
        ):
    
    vectorizer = TfidfVectorizer(
        ngram_range=ngram,
        max_features=max_features, 
        stop_words=stop_words
    )

    X = vectorizer.fit_transform(text)

    result = pd.DataFrame({
        'Term': vectorizer.get_feature_names_out(),
        'Score': X.sum(axis=0).A1
    }).sort_values(
        'Score',
        ascending=False
    )

    return result

# Get top Negative keywords from cleaned review (tokenized review)
# %%
# Get the processed_texts column of the negative reviews
negative_reviews = df_clean.loc[
    df_clean['sentiment']=='Negative', 
    'processed_texts'
    ]

# Confirm creation of series of negative_reviews 
print(negative_reviews.head(10))

# %%
# Identify top 20 negative keywords from the negative texts column
negative_keywords = get_ngram_terms(
    negative_reviews, 
    ngram=(1,1), 
    max_features=20
    )

print(negative_keywords)

# %%
# Visualise negative keywords distribution
negative_keywords_plot = negative_keywords.sort_values(
    'Score',
    ascending=True
)

negative_keywords_plot.plot(
    x='Term',
    y='Score',
    kind='barh',
    figsize=(10, 8),
    legend=False
)

plt.title('Top 20 Negative Keywords')
plt.xlabel('Score')
plt.ylabel('Keyword')
plt.tight_layout()
plt.show()


# Get top Positive keywords from cleaned review (tokenized review)
# %%
# Get processed_text column of the positive reviews
positive_reviews = df_clean.loc[df_clean['sentiment']=='Positive',
                                'processed_texts']

# Identify top 20 positive keywords from the negative texts column
positive_keywords = get_ngram_terms(
    positive_reviews, 
    ngram=(1,1), 
    max_features=20
    )

print(positive_keywords)

# %%
# Visualize positive reviews distribution

positive_keywords_plot = positive_keywords.sort_values(
    'Score',
    ascending=True
)

positive_keywords_plot.plot(
    x='Term',
    y='Score',
    kind='barh',
    figsize=(10, 8),
    legend=False
)

plt.title('Top 20 Positive Keywords')
plt.xlabel('Score')
plt.ylabel('Keyword')
plt.tight_layout
plt.show()


# %%
# Bigram Analysis
#-----------------
# Generate review bigrams across all the reviews
df_bigram = get_ngram_terms(
    df_clean['processed_texts'], 
    ngram=(2,2), 
    max_features=20
    )

print(df_bigram)

# %%
# Generate negative review bigrams
df_negative_bigram = get_ngram_terms(
    negative_reviews, 
    ngram=(2,2), 
    max_features=20
    )

print(df_negative_bigram)

# %%
# Add domain specific stopwords as some of these words are dominating the bigram result
domain_stop_words = [
    'amazon',
    'review',
    'text',
    'find',
    'will',
    'not'
    ]

# Rerun bigram analysis
df_bigram_repeat = get_ngram_terms(
    df_clean['processed_texts'],
    ngram=(2,2),
    max_features=20,
    stop_words=domain_stop_words
)

print(df_bigram_repeat)

# %%
# Rerun bigram for negative reviews
df_bigram_negative = get_ngram_terms(
    negative_reviews,
    ngram=(2,2),
    max_features=20,
    stop_words=domain_stop_words
)

print(df_bigram_negative)

# %%
# Visualise the top 10 negative review bigrams
negative_bigram_top10 = df_bigram_negative.head(10)
negative_bigram_top10_plot = negative_bigram_top10.sort_values(
    'Score',
    ascending=True
)

negative_bigram_top10_plot.plot(
    x='Term',
    y='Score',
    kind='barh',
    figsize=(10,8),
    legend=False
)

plt.title('Top 10 Negative Review Bigram')
plt.xlabel('Score')
plt.ylabel('Terms')
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------
# Sentiment distribution
#----------------------
# Sentiment distribution is heavily skewed towards negative reviews, with 68.2% of reviews classified as 
# negative. Positive reviews account for 27.6%, while neutral reviews represent only 4.2% of the dataset, 
# indicating a strong imbalance in customer sentiment.
# 
# Home & Living recorded the highest negative sentiment (70.0%).
# Fashion recorded the highest positive sentiment (29.4%).
# Negative sentiment exceeded 66% in every category.
# The variation across categories is relatively small, suggesting customer dissatisfaction is a platform-wide 
# issue rather than being concentrated in a specific product category.
#
# There are a total of 19128 review records for the top 10 countries by volume
# the top 10 countries are US, GB, CA, IN, IE, DK, NL, AU, DE, IT. 
# Total reviews from this country makes up 90.86% of the total reviews
# US and GB account for 78.8% of all reviews
# Canada has the highest negative sentiment (77.7%)
# Italy has the highest positive sentiment (56.0%)
# Germany also shows relatively balanced sentiment (45.9% positive)
# Most major markets exhibit predominantly negative sentiment

# Ngram analysis
#----------------
# Tf-IDF analysis identified customer, service, order, delivery, refund, account, return and prime among 
# the most influential keywords in negative reviews. Generic high-frequency terms such as amazon, not, day, 
# time, company and get were also ranked highly. Overall, the results indicate that negative reviews are 
# characterised by a mixture of service-related, transaction-related and platform-related vocabulary.
#
# TF-IDF analysis of positive reviews identified good, great, love, service, delivery, price, prime, and 
# customer among the most influential keywords. Generic terms such as amazon, review, company, time, and 
# day were also ranked highly due to their frequent occurrence across positive reviews.
# Overall, positive reviews are characterised by favourable evaluations of service quality, delivery 
# experience, product value, and the overall shopping experience.
#
# Bigram analysis results were dominated by phrases containing domain_specific and generic high-frequency 
# words such as 'amazon', 'review', 'text', 'find', 'will', and 'not', resulting in bigrams such as 'amazon
# prime', 'amazon customer', 'will not', and 'not know' that provided limited business insight.
# 
# After removal of stopwords such as will, not, amazon, text, find, review, not, the bigram result became
# substantially more informative, revealing revealing recurring phrases related to customer serviec, 
# deliveries, orders, returns, account management, Prime membership, and payment methods.
#
# The most prominent phrase was 'customer service', followed by phrases such as 'day delivery', 'delivery 
# driver', 'cancel order', 'return item', 'close account', 'prime membership', 'credit card', and 'gift
# card'. Based on these results, the major themes identified were Customer Service & Support, Delivery &
# Fulfilment, Order Management, Returns & Account Management, and Prime Membership & Payments.
#-----------------------------------------------------------------------------------------------------------
