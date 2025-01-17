# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:58:34 2025

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

# 1. Business Problem
# Objective: Predict if a tweet is about a real disaster (1) or fake disaster (0)
# Constraints: The model should be able to handle text data and predict effectively.

# 2. Data Pre-processing
# Load the dataset
file_path = 'E:/Honars(DS)/Data Science/Naive Bayes Assignment/Disaster_tweets_NB.csv'
df = pd.read_csv(file_path)

# Display first few rows of the dataset to understand its structure
print("Dataset Overview:")
print(df.head())

# 2.1 Data Dictionary (Example)
# Feature | Data Type | Relevance to Model
# -----------------------------------------
# id      | Numeric   | Irrelevant (no predictive power, can be dropped)
# message | Text      | Relevant (primary input for prediction)
# label   | Categorical (1, 0) | Relevant (target variable, 1 = real, 0 = fake)
# You can continue this analysis for all features.

# 3. Data Cleaning and Feature Engineering
# Handle missing values (if any)
df = df.dropna(subset=['message', 'label'])  # Drop rows with missing text or label

# Encode the label (1 = real disaster, 0 = fake disaster)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Text preprocessing (Tokenization, stopword removal, etc.)
# Convert messages to lowercase and remove non-alphabetic characters
df['message'] = df['message'].str.lower().str.replace(r'[^a-z\s]', '', regex=True)

# 4. Exploratory Data Analysis (EDA)
# Summary of the dataset
print("Summary of Dataset:")
print(df.describe())

# Univariate Analysis (Label distribution)
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Distribution of Real vs Fake Tweets')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Fake', 'Real'])
plt.show()

# Bivariate Analysis (Message Length vs Label)
df['message_length'] = df['message'].apply(len)
plt.figure(figsize=(8, 6))
sns.boxplot(x='label', y='message_length', data=df)
plt.title('Message Length vs Label')
plt.xlabel('Label')
plt.ylabel('Message Length')
plt.xticks(ticks=[0, 1], labels=['Fake', 'Real'])
plt.show()

# 5. Model Building
# Feature Extraction: Convert text data into numerical format using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)  # Limit to 5000 most frequent words
X = vectorizer.fit_transform(df['message']).toarray()  # Convert text to bag-of-words

# Target variable
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5.1 Build Naïve Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test)

# 5.2 Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5.3 Tune the Model (Example: Varying the Number of Features)
vectorizer = CountVectorizer(max_features=10000)  # Try using more features
X = vectorizer.fit_transform(df['message']).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Rebuild and evaluate Naïve Bayes Model
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after tuning (more features): {accuracy:.4f}")

# Confusion Matrix after tuning
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix after Tuning:")
print(conf_matrix)

# 6. Benefits/Impact of the Solution
# This solution helps businesses in various sectors, such as media or emergency response, to detect fake disaster-related information.
# By predicting whether a tweet is about a real or fake disaster, it helps mitigate misinformation.
# This can lead to more efficient communication, timely responses, and better decision-making for authorities.
print("""
Benefits/Impact:
- Helps detect fake disaster tweets, improving response and resource allocation during real disasters.
- Enables efficient monitoring of social media for real-time disaster information.
- Supports the creation of systems that can automatically alert authorities to fake disaster reports, reducing misinformation risks.
""")
