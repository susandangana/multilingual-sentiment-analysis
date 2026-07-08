#=======================================================================================================================================
# Library Import
#=======================================================================================================================================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data_preprocessing_utils import process_text_for_prediction
#=======================================================================================================================================
# Page Configuration
#=======================================================================================================================================
st.set_page_config(
    page_title='ShopEase Europe Sentiment Analyser',
    page_icon='📊',
    layout='wide'
)

#=======================================================================================================================================
# Load Saved Model and Vectorizer
#=======================================================================================================================================
model = joblib.load('models/logistic_regression.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

#=======================================================================================================================================
# Side Bar
#=======================================================================================================================================
st.sidebar.header('About')
st.sidebar.info("""
### About This Application

**ShopEase Europe – Multilingual Customer Sentiment Analysis**

This application predicts the sentiment of e-commerce customer reviews using a machine learning model trained on multilingual customer feedback.

### Model

- **Algorithm:** Logistic Regression
- **Text Representation:** TF-IDF Vectorization
- **Sentiment Classes:** Positive, Neutral and Negative

### Dataset

- Approximately **21,000** customer reviews
- Reviews collected from multiple countries
- Primary language: English
- Additional languages detected during preprocessing: Italian, French, German, Spanish, Dutch, Polish and Danish

### Text Processing Pipeline

- Text cleaning and normalization
- URL, email, punctuation and emoji removal
- Automatic language detection
- Language-specific tokenization
- Stop-word removal
- Lemmatization
- TF-IDF feature extraction
- Sentiment prediction using Logistic Regression

### Application Features

- Predict sentiment for a single customer review
- Perform batch prediction using a CSV file
- Visualize predicted sentiment distribution
- Download prediction results as CSV

### Limitations

- Performance may vary for languages with limited training examples.
- Neutral reviews are more difficult to classify due to class imbalance.
- Sarcasm, figurative language and highly contextual reviews may not always be interpreted correctly.

### Repository

GitHub: *https://github.com/susandangana/multilingual-sentiment-analysis*
Supported sentiment classes:
- Positive
- Neutral
- Negative
""")

#=======================================================================================================================================
# App Title and Description
#=======================================================================================================================================
st.title('📊 Customer Sentiment Analyzer')

st.markdown("""
Predict ShopEase customer sentiment from e-commerce reviews using a machine learning 
model trained on ShopEase Europe customer feedback.
""")

#=======================================================================================================================================
# Project Metrics
#=======================================================================================================================================
metric1, metric2 = st.columns(2)

with metric1:
    st.metric('Model', 'Logistic Regression')

with metric2:
    st.metric('Prediction Classes', '3 Classes')

st.divider()


#=======================================================================================================================================
# Input Section
#=======================================================================================================================================
col1, col2 = st.columns([3,2])

with col1:
    st.header('Single Review Prediction')
    
    review = st.text_area(
        'Enter a customer review:',
        height=150
        )

    predict_single = st.button('Predict Sentiment')

with col2:
    st.header('Batch Prediction (CSV Upload)')

    uploaded_file = st.file_uploader(
        'Upload a CSV file containing a review column',
        type=['csv']
        )
   
    run_batch = False

    if uploaded_file is not None:
        run_batch = st.button('Run Batch Prediction')

#=======================================================================================================================================
# Single Prediction Output
#=======================================================================================================================================
if predict_single:
    
    if review.strip() == '':
        st.warning('Please enter a review.')

    else:
            
        # Apply same preprocessing used during training
        processed_review = process_text_for_prediction(review)

        review_vector = vectorizer.transform([processed_review])

        prediction = model.predict(review_vector)[0]

        st.subheader('Prediction')
        
        if prediction == 'Positive':
            st.success(f'😊 Predicted Sentiment: {prediction}')

        elif prediction =='Negative':
            st.error(f'☹️ Predicted Sentiment: {prediction}')

        else:
            st.info(f'😐 Predicted Sentiment: {prediction}')

#========================================================================================================================================
# Batch Prediction Output
#========================================================================================================================================
if uploaded_file is not None and run_batch:
    
    df = pd.read_csv(uploaded_file)

    if 'review' not in df.columns:
        
        st.error("CSV must contain a column named 'review'")
    
    else:

        with st.spinner('Generating predictions...'):

            # Hadle missing reviews
            df['review'] = df['review'].fillna('')
        
            # Apply same preprocessing used during training
            df['processed_texts'] = (df['review']
                                     .apply(process_text_for_prediction)
                                     )

            review_vectors = vectorizer.transform(df['processed_texts'])

            df['predicted_sentiment'] = predictions = model.predict(
                review_vectors
                )
            
        st.success('Batch prediction completed successfully.')
        st.write(f'**Total Reviews Processed:** {len(df):,}')

        st.divider()

        st.subheader('Preview')
        st.dataframe(df.head(), use_container_width=True)

        st.subheader('Prediction Results')
        st.dataframe(
                    df[['review', 'predicted_sentiment']],
                    use_container_width=True
                    )
        
        # Sentiment Distribution Chart
        st.subheader('Sentiment Distribution')

        sentiment_counts = (
            df['predicted_sentiment']
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots()

        sentiment_counts.plot(
            kind='bar',
            ax=ax
            )

        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_title('Predicted Sentiment Distribution')
        plt.xticks(rotation=0)

        st.pyplot(fig)

        # Download Results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download Predictions',
            data=csv,
            file_name='sentiment_predictions.csv',
            mime='text/csv'
            )
            
#==============================================
# Footer
#==============================================
st.divider()
st.caption(
    'Built by Susan Dangana using Python, spaCy, Scikit-learn and Streamlit'
    )
        
    