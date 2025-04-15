# Import necessary libraries
import pandas as pd  # For data manipulation (although it's not used directly here, it may be useful for future expansion)
import pickle as pk  # For loading the trained model and scaler (vectorizer)
from sklearn.feature_extraction.text import TfidfVectorizer  # Import the vectorizer (although it's not directly used here)
import streamlit as st  # Streamlit for creating the interactive web app

# Load the pre-trained model and the scaler (TfidfVectorizer)
model = pk.load(open('model.pkl', 'rb'))  # Load the trained Logistic Regression model from 'model.pkl'
scaler = pk.load(open('scaler.pkl', 'rb'))  # Load the trained TfidfVectorizer from 'scaler.pkl'

# Create a text input field in the Streamlit app for entering a movie review
review = st.text_input('Enter Movie Review')  # Prompt the user to input a review

# When the "Predict" button is clicked, predict the sentiment of the entered review
if st.button('Predict'):  # Check if the 'Predict' button is clicked
    # Transform the entered review using the loaded scaler (TfidfVectorizer) to match the model's training format
    review_scale = scaler.transform([review]).toarray()  # Convert the review text into a numerical format
    result = model.predict(review_scale)  # Use the trained model to predict the sentiment of the review

    # Display the sentiment result: Negative (0) or Positive (1)
    if result[0] == 0:  # If the result is 0 (Negative sentiment)
        st.write('Negative Review')  # Display 'Negative Review'
    else:  # If the result is 1 (Positive sentiment)
        st.write('Positive Review')  # Display 'Positive Review'
