# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:51:02 2024

@author: Dell
"""

import requests  # Used for sending HTTP requests to fetch the webpage content
from bs4 import BeautifulSoup  # Used for parsing HTML content and extracting data
from textblob import TextBlob  # Used for performing sentiment analysis on text

# Function to clean the review text (removes extra spaces, new lines, etc.)
# This ensures the reviews are formatted properly without unnecessary whitespaces or line breaks.
def clean_text(text):
    return ' '.join(text.split())

# Example IMDB movie review page link (The Shawshank Redemption reviews)
# This is the URL for the IMDb reviews page for the movie "The Shawshank Redemption."
url = "https://www.imdb.com/title/tt0111161/reviews/?ref_=tt_ov_ql_2"

# Send a GET request to the IMDb reviews page
# This fetches the HTML content of the IMDb reviews page.
response = requests.get(url)

# Parse the page content using BeautifulSoup
# BeautifulSoup helps in parsing the HTML content fetched from IMDb.
soup = BeautifulSoup(response.content, 'html.parser')

# Extract reviews from the HTML page
# Finds all <div> elements that have the class 'text show-more__control' which contains the review text.
reviews = soup.find_all('div', class_='text show-more__control')

# Print the first 5 reviews (before cleaning)
print("Original Reviews:\n")
for i, review in enumerate(reviews[:5], 1):  # Loop through the first 5 reviews
    print(f"Review {i}: {review.get_text(strip=True)}\n")  # Print the review content, stripping extra spaces

# Clean the reviews
# Applies the clean_text function to remove extra whitespaces from the reviews.
cleaned_reviews = [clean_text(review.get_text(strip=True)) for review in reviews]

# Perform sentiment analysis on the cleaned reviews
# Now that we have clean reviews, we use TextBlob to analyze the sentiment for each review.
print("\nSentiment Analysis on the First 5 Reviews:\n")
for i, review in enumerate(cleaned_reviews[:5], 1):  # Analyze the first 5 cleaned reviews
    # Calculate the sentiment polarity using TextBlob
    # Polarity ranges from -1 (negative) to 1 (positive); closer to 0 indicates neutral.
    sentiment_polarity = TextBlob(review).sentiment.polarity
    
    print(f"Review {i}: Sentiment Polarity = {sentiment_polarity}")  # Output the polarity score
    
    # Classify the sentiment based on the polarity value
    if sentiment_polarity > 0:
        print("Sentiment: Positive\n")  # Positive sentiment
    elif sentiment_polarity < 0:
        print("Sentiment: Negative\n")  # Negative sentiment
    else:
        print("Sentiment: Neutral\n")   # Neutral sentiment
