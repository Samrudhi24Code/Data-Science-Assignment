# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:00:18 2024

@author: Dell
"""

# Problem Statement:
# The task is to extract customer reviews for a product from a Meesho product page and perform sentiment analysis on those reviews.
# We will fetch the reviews from the Meesho product page, clean the extracted text, and perform sentiment analysis to classify the sentiment of each review as positive, negative, or neutral based on the sentiment polarity.

# Business Objective:
# Sentiment analysis of product reviews helps in understanding customer opinions and feedback about a product. 
# This can be leveraged by businesses to evaluate the success of a product, detect areas for improvement, and guide future product development or marketing strategies.
# For instance, a large number of negative reviews might indicate the need for product improvement or a change in its marketing strategy.

# Solution:
# The solution fetches product reviews from a Meesho product page using web scraping. The reviews are then cleaned using regular expressions to remove unnecessary characters. 
# After that, sentiment analysis is performed using the TextBlob library to classify each review's sentiment. Based on the sentiment polarity score, reviews are categorized into positive, negative, or neutral.

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
