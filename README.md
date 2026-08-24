# Hotel Review Rating Prediction using Naive Bayes [ project done in 1st year] 

## Run the frontend

Install the dependencies and launch the Streamlit interface:

```bash
pip install -r requirements.txt
streamlit run app.py
```
first opening page:
<img width="2059" height="1228" alt="image" src="https://github.com/user-attachments/assets/5f347c36-e865-4cc3-920f-7602592521ec" />

Predefined comments/complains:
<img width="1969" height="1145" alt="image" src="https://github.com/user-attachments/assets/7bad46ad-48c8-4e09-88fb-1f4d3ea466dd" />

user customised complaint: review : good
<img width="2056" height="1122" alt="image" src="https://github.com/user-attachments/assets/986d35b2-9772-461e-8bc9-5719246d0eef" />
user customised complaint: review : bad
<img width="2063" height="1175" alt="image" src="https://github.com/user-attachments/assets/0f5723c2-62d1-48b8-9734-de0d5dc57d8a" />

dataset signals:
<img width="2065" height="1152" alt="image" src="https://github.com/user-attachments/assets/7c14e685-aa1f-4814-bbeb-1a1a8bebca9b" />

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
