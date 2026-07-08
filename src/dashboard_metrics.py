
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
import csv

#======================================================================================================


#==========================================================================================================
# %%
# Dashboard-Ready Metrics
#==========================================================================================================

# Load clean dataset
df_clean = pd.read_csv('../data/processed_reviews.csv')


# Dashboard KPIs
dashboard_kpis = pd.DataFrame({
    'KPI': [
        'Total Reviews',
        'Average Rating'
    ],
    'Value': [len(df_clean),
              round(df_clean['rating'].mean(),2)
              ]
})

# Save to csv
dashboard_kpis.to_csv(
    '../outputs/dashboard/dashboard_kpis.csv',
    index=False
)


# Setiment Summary 
#----------------------
sentiment_summary = (
    df_clean['sentiment']
    .value_counts()
    .reset_index()
)

sentiment_summary.columns = [
    'Sentiment',
    'Review_Count'
]

sentiment_summary['Percentage'] = (
    sentiment_summary['Review_Count']
    / sentiment_summary['Review_Count'].sum()
    * 100
).round(2)

print(sentiment_summary)

# Save to csv
sentiment_summary.to_csv(
    '../outputs/dashboard/sentiment_summary.csv',
    index=False
)

# Reviews by Country 
#----------------------
country_summary = (
    df_clean['country']
    .value_counts()
    .reset_index()
)

country_summary.columns = [
    'Country',
    'Review_Count'
]

print(country_summary)

# Save to csv
country_summary.to_csv(
    '../outputs/dashboard/country_summary.csv',
    index=False
)
# %%
# View timestamp datatype
print(df_clean['timestamp'].dtype)

# Convert the timestamp from string to datetime
df_clean['timestamp'] = pd.to_datetime(
    df_clean['timestamp'],
    format='mixed',
    utc=True
)

# Remove timezone
df_clean['timestamp'] = (
    df_clean['timestamp']
    .dt.tz_localize(None)
)

#%%
print(df_clean['timestamp'].head())
# %%
#  Monthly Review Trend
#----------------------
monthly_reviews = (
    df_clean
    .groupby(
        df_clean['timestamp'].dt.to_period('M')
    )
    .size()
    .reset_index(name='Review_Count')
)

print(monthly_reviews)

# Save to csv
monthly_reviews.to_csv(
    '../outputs/dashboard/monthly_reviews.csv',
    index=False
)
# %%
# Complaint Category (based on the negative theme analysis)
#-------------------------
complaint_summary = pd.DataFrame({
    'Complaint_Category': [
        'Customer Service',
        'Delivery Issues',
        'Order & Returns',
        'Prime Membership'
    ]
})

print(complaint_summary)

# Save to csv
complaint_summary.to_csv(
    '../outputs/dashboard/complaint_summary_csv',
    index=False
)

# Satisfaction Drivers (based on the positive theme analysis)
#----------------------------
satisfaction_summary = pd.DataFrame({
    'Satisfaction_Driver': [
        'Service Quality',
        'Delivery Experience',
        'Product Quality',
        'Value for Money',
        'Prime Benefits'
    ]
})

print(satisfaction_summary)

# Save to csv
satisfaction_summary.to_csv(
    '../outputs/dashboard/satisfaction_summary',
    index=False
)

# Create dashboard specific dataset by dropping the processed_token, processed_texts and other columns not 
# relevant to building a dashboard
# Reason being that processed tokens contains lists and powerbi sometimes has issues importing complex text
# representation. Also processed texts column contains null value as seen earlier and since these columns 
# are not needed in for the dashboard, it is safe to remove them before exporting to PowerBI
dashboard_cols = df_clean.columns.to_list()

df_dashboard = df_clean.drop(
    columns=['processed_tokens', 'processed_texts', 'language', 'review', 'review_lower', 'clean_review'],
    errors='ignore'
)

# Save dashboard dataset
df_dashboard.to_csv(
    '../data/dashboard_dataset.csv',
    index=False,
    encoding='utf-8',
    quoting=csv.QUOTE_ALL
)

# View dataset
print(df_dashboard.head())

# %%

