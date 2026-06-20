# E-commerce Sentiment Analysis
## Project Overview

This project develops a multilingual sentiment analysis solution for customer reviews collected from an e-commerce platform. The objective is to automatically classify customer feedback as positive, neutral, or negative and identify key drivers of customer satisfaction and dissatisfaction.

## Business Problem
As e-commerce business grow, manually reviewing thousands of customer comments becomes impractical. Automated sentiment analysis enables organisationa to:

- Monitor customer satisfaction at scale
- Identify recurring customer issues
- Detect emerging trends in feedback
- Support data-driven business decisions
- Improve products and customer experience

## Dataset Availability

The original dataset contains 120,000 customer reviews. To keep the repository lightweight and focused on the analysis workflow, the raw and processed datasets are not included in this repository.

To reproduce the analysis, place the source dataset in the data/ directory and update the file paths in the project code as required. 

The dataset contains customer reviews across multiple countries and languages, including English, French, German, and Spanish.It contains the following features:

| Feature | Description |
|----------|------------|
| review_id | Unique identifier for each review |
| product_category | Product category reviewed |
|timestamp | Date the review was submitted |
| country | Customer country code |
| rating | Customer rating (1-5)
| review | Customer review text |
| sentiment | Sentiment label (Positive, Neutral, Negative)

## Project Workflow
- Data Quality Assessment
- Data Cleaning
- Multilingual Language Detection
- Text Preprocessing
- Exploratory Data Analysis
- Sentiment Classification
- Topic Modelling
- Dashboard Development

## Technologies
- Python
- Pandas
- NumPy
- spaCy
- ftlangdetect
- clean-text
- Scikit-learn
- Matplotlib
- Seaborn


## Repository Structure

ecommerce-sentiment-analysis/
│
├── data/
│   ├── raw_review.csv 
│   └── processed_review.csv
│   
├── src/
│   └── customer_sentiment_analysis.py
│   
├── README.md
├── requirements.txt
└── .gitignore
