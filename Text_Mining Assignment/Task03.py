# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:00:18 2024

@author: Dell
"""

# Import required libraries
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import re

# Function to clean the text by removing unnecessary characters
def clean_text(text):
    return re.sub(r'[^A-Za-z\s]', '', text)

# Review page link
url = "https://www.meesho.com/zara/p/67epva"  # Meesho product page

# Send a GET request to the page
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract reviews (adjust class as needed)
    reviews = soup.find_all('div', class_='review-text')

    # Example: Print the first 5 extracted reviews
    for i, review in enumerate(reviews[:5], 1):
        print(f"Review {i}: {review.get_text(strip=True)}\n")

    # Clean the reviews using the `clean_text` function
    cleaned_reviews = [clean_text(review.get_text(strip=True)) for review in reviews]
    print("Cleaned Reviews:", cleaned_reviews[:5])  # Print cleaned versions of the first 5

    # Perform sentiment analysis on the cleaned reviews
    print("\nSentiment Analysis Results:")
    for i, review in enumerate(cleaned_reviews[:5], 1):  
        sentiment = TextBlob(review).sentiment.polarity
        print(f"Review {i}: Sentiment Polarity = {sentiment}")
        if sentiment > 0:
            print("Sentiment: Positive\n")
        elif sentiment < 0:
            print("Sentiment: Negative\n")
        else:
            print("Sentiment: Neutral\n")
else:
    print(f"Failed to retrieve page. Status code: {response.status_code}")
