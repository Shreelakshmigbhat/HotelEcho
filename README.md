# Hotel Review Rating Prediction using Naive Bayes

## Run the frontend

Install the dependencies and launch the Streamlit interface:

```bash
pip install -r requirements.txt
streamlit run app.py
```
<img width="2554" height="1423" alt="image" src="https://github.com/user-attachments/assets/c79295ca-2645-438d-88fa-5a28a66e4df9" />


The app loads `tripadvisor_hotel_reviews.csv` from the project folder, trains the model on startup, and provides review predictions plus dataset insights.

dataset used - https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews


#Overview

This project implements a machine learning model to predict hotel ratings based on customer text reviews.
It uses a Naive Bayes classifier with CountVectorizer to process text data and classify the sentiment or rating level.

Dataset

File: tripadvisor_hotel_reviews.csv

Columns:

Review: Text of the hotel review.

Rating: Numerical rating provided by the reviewer (e.g., 1–5).

Methodology

Data Loading
The dataset is loaded using the pandas library.

Feature Extraction

Reviews (X) and Ratings (y) are extracted using iloc.

Text data is transformed into numerical feature vectors using CountVectorizer.

Data Splitting
The dataset is divided into training and testing sets using an 80:20 ratio with train_test_split.

Model Training
A Multinomial Naive Bayes classifier is trained on the vectorized data.

Evaluation
The model’s accuracy is calculated using accuracy_score from sklearn.metrics.

Prediction Function
A custom function predict_rating() allows users to input a review and receive a predicted rating.

Future Enhancements

Implement TF-IDF vectorization for improved text representation.

Apply advanced models such as Logistic Regression or Neural Networks.

Add preprocessing steps such as stopword removal and lemmatization.

Develop a web-based user interface using Flask or Streamlit.
