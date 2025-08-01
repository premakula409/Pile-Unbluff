# Install necessary libraries
!pip install gensim
!pip install gradio

# Import necessary libraries
import pandas as pd
import numpy as np
from google.colab import drive
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import re
from nltk.corpus import stopwords
import nltk

# Mount Google Drive to access the dataset
drive.mount('/content/drive')

# Load the dataset from Google Drive
file_path = "/content/drive/MyDrive/Colab Notebooks/Smart_Library_Dataset.csv"
df = pd.read_csv(file_path)

# Download NLTK stop words for text cleaning
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a function to clean the text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop words
    return text

# Combine 'Book Name' and 'Keywords' into 'Text' and preprocess it
df['Text'] = df['Book Name'] + " " + df['Keywords']
df['Text'] = df['Text'].apply(preprocess_text)

# Load pre-trained Word2Vec model
print("Loading Word2Vec model... This may take a moment.")
wv = api.load('word2vec-google-news-300')
print("Model loaded successfully.")

# Get average word embedding for a given text
def get_average_embedding(text):
    words = text.split()
    embeddings = [wv[word] for word in words if word in wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(300)

# Compute embeddings for each book
book_embeddings = np.array([get_average_embedding(text) for text in df['Text']])

# Book recommendation function
def recommend_books(query, n=3, threshold=0.1):
    query = preprocess_text(query)
    query_embedding = get_average_embedding(query)
    if np.all(query_embedding == 0):
        return "No relevant books found."
    similarities = cosine_similarity([query_embedding], book_embeddings)[0]
    top_indices = similarities.argsort()[-n:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            book = df.iloc[idx]
            results.append(f"{book['Book Name']} (Rack {book['Rack']}, Row {book['Row']})")
    if results:
        return "\n".join(results)
    else:
        return "No relevant books found."

# Gradio interface
gr.Interface(
    fn=recommend_books,
    inputs="text",
    outputs="text",
    title="Smart Library Book Finder - PileUnbluff",
    description="Enter keywords to find books hidden in a pile!"
).launch()
##  Demo Interface

![Gradio Demo](assets/demo.png)
