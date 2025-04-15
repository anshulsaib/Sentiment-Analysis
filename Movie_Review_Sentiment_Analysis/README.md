Sentiment Analysis of Movie Reviews
This project uses machine learning to classify movie reviews as positive or negative based on the content of the review. It employs natural language processing (NLP) techniques to clean and process the text, and a logistic regression model to predict sentiment. The project also includes an interactive web app built using Streamlit to allow users to enter their own movie review and get predictions.

Technologies Used
Python - Programming language used for the project

pandas - Used for data manipulation and cleaning

nltk - Natural language processing library used for text preprocessing (stopword removal)

scikit-learn - Used for machine learning, feature extraction (TfidfVectorizer), and model training (Logistic Regression)

Streamlit - Used to build the interactive web app

WordCloud - To generate word clouds based on the movie review data

Project Structure
Movie_Review.csv - Contains the dataset with movie reviews and their respective sentiments.

File-1.py - Script for cleaning the dataset, training the logistic regression model, and creating visualizations like word clouds.

app.py - Streamlit web app that allows users to input a movie review and predicts whether the review is positive or negative.

model.pkl - The trained logistic regression model (serialized using pickle).

scaler.pkl - The trained TfidfVectorizer (serialized using pickle).

Setup
Prerequisites
Make sure you have Python 3.6+ installed on your machine. You will also need the following libraries:

pandas

matplotlib

nltk

scikit-learn

streamlit

wordcloud

You can install the required libraries using pip:

bash
Copy
Edit
pip install pandas matplotlib nltk scikit-learn streamlit wordcloud
Running the Project
Step 1: Prepare the dataset

Place your Movie_Review.csv file in the same directory as the project files. This file should contain movie reviews with a column for sentiment (pos or neg).

Step 2: Train the model

Run File-1.py to clean the dataset, create word clouds, and train the logistic regression model on the dataset. The trained model and scaler will be saved as model.pkl and scaler.pkl.

bash
Copy
Edit
python File-1.py
Step 3: Run the Streamlit web app

Once the model is trained, run the app.py file to start the interactive web app where you can input movie reviews and predict their sentiment.

bash
Copy
Edit
streamlit run app.py
Step 4: Use the app

Open the app in your browser (the URL will be displayed in the terminal after running the above command).

Enter a movie review in the text input box and click "Predict" to get the sentiment of the review (positive or negative).

How It Works
Text Preprocessing:

The text from the movie reviews is cleaned by removing common stopwords (like "the", "a", etc.) using NLTK.

Feature Extraction:

The cleaned reviews are converted into numerical data using TfidfVectorizer. This technique converts text into a matrix of numbers, where each word's importance is based on its frequency in a document and across all documents.

Model Training:

A Logistic Regression model is trained on the processed data to classify reviews into positive or negative sentiments.

Predictions:

The trained model is used to predict the sentiment of new movie reviews entered via the Streamlit app.

Visualization:

Word clouds are generated to visualize the most frequent words in positive and negative reviews.

Example Output
Input: "The movie was fantastic, great plot and acting!"

Output: "Positive Review"

Input: "I hated the movie, it was a waste of time."

Output: "Negative Review"

Future Improvements
Incorporating more advanced machine learning models such as Random Forest or Neural Networks for better accuracy.

Expanding the dataset to improve model generalization.

Adding more advanced text preprocessing techniques like lemmatization and stemming.

License
This project is licensed under the MIT License - see the LICENSE file for details.

