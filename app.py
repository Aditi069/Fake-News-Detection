import base64
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

app = Flask(__name__)

# Load the model
lr = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Load the fitted TF-IDF vectorizer
with open('fitted_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to make predictions
def predict(input_text):
    # Preprocess the input text
    processed_text = preprocess_text(input_text)

    # Vectorize the processed text using the loaded vectorizer
    input_vector = vectorizer.transform([processed_text])

    # Make prediction
    prediction = lr.predict(input_vector)[0]
    return "Fake News" if prediction == 0 else "True News"

# Function to generate and return Word Cloud image as a base64-encoded string
def generate_wordcloud_image(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Convert the Word Cloud image to a base64-encoded string
    img_buffer = BytesIO()
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    wordcloud_img = None

    if request.method == 'POST':
        input_text = request.form['input_text']
        result = predict(input_text)
        # Generate Word Cloud for the input text
        wordcloud_img = generate_wordcloud_image(input_text)

    return render_template('index.html', result=result, wordcloud_img=wordcloud_img)

if __name__ == '__main__':
    app.run(debug=True)
