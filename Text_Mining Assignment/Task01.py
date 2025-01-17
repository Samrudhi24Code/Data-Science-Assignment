# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:00:18 2024

@author: Dell
"""


'''Objective:
    Need to understand customer feedback by analyzing reviews using text
    mining and sentiment analysis. 
'''
##############################################################
'''
Task 1:
    1.	Extract reviews of any product from e-commerce website Amazon.
    2.	Perform sentiment analysis on this extracted data and build
    aunigram and bigram word cloud.'''



# Import required libraries

# requests: Used for sending HTTP requests to fetch web pages
import requests

# BeautifulSoup: Used for parsing HTML and extracting specific elements from the page
from bs4 import BeautifulSoup as bs

# re: Regular expressions for cleaning and manipulating text
import re

# nltk: Natural Language Toolkit for text processing and stopword removal
import nltk
from nltk.corpus import stopwords

# TextBlob: Simplified text processing library for sentiment analysis
from textblob import TextBlob

# WordCloud: Used for generating word clouds from text data
from wordcloud import WordCloud

# matplotlib: For plotting and displaying the word cloud
import matplotlib.pyplot as plt

#########################################

# Define the link to the specific Amazon review
link = "https://www.amazon.com/product-reviews/B01AMT0EYU"

# Set headers to make the request look like it’s coming from a regular browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}

# Fetch the page with headers
page = requests.get(link, headers=headers)

# Check if the connection is successfully established (status code 200)
if page.status_code == 200:
    print("Connection Successful!")
else:
    print(f"Failed to connect. Status code: {page.status_code}")

# Extract the content of the page
page_content = page.content

# Check if the page content is properly fetched
print(f"Page Status Code: {page.status_code}")
print("First 500 characters of the page content:\n", page_content[:500], "\n")

# Parse the page content using BeautifulSoup and HTML parser
soup = bs(page_content, 'html.parser')

#########################################

# Extract the Actual Review Text

# Attempt to extract review text using the proper HTML tag and class (found by inspecting the page)
try:
    review = soup.find('span', {'data-hook': 'review-body'}).get_text(strip=True)
    print("Review Text:", review)
except AttributeError:
    print("Review text not found. Please inspect the page structure again.")

#########################################

# Text Cleaning Process

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Function to clean the review text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords (common words like 'the', 'is', etc.)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Clean the extracted review text (if present)
if 'review' in locals():
    review_cleaned = clean_text(review)
    print("Cleaned Review:", review_cleaned)
else:
    review_cleaned = ""
    print("No review text extracted.")

#########################################

# Sentiment Analysis with TextBlob

# Function to perform sentiment analysis using TextBlob
def get_sentiment(text):
    # Create a TextBlob object and analyze polarity (-1 to 1)
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    # Classify sentiment based on polarity score
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Perform sentiment analysis on cleaned review
if review_cleaned:
    sentiment = get_sentiment(review_cleaned)
    print("Sentiment of the Review:", sentiment)
else:
    print("No text available for sentiment analysis.")

#########################################

# Generate a Word Cloud from the Review Text

# Function to generate a unigram word cloud
def generate_unigram_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axis lines
    plt.show()

# Generate a word cloud for the cleaned review
if review_cleaned:
    generate_unigram_wordcloud(review_cleaned)
else:
    print("No cleaned review available for word cloud generation.")
