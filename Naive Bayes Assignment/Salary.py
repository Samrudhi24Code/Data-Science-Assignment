# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:41:50 2025
@author: Dell
"""

'''
1. Business Problem
Objective:

The goal is to classify employees into salary brackets (e.g., "<=50K" or ">50K") using the provided salary dataset.
The objective is to assist organizations in analyzing salary distributions and predicting the salary category for new employees based on features like education, occupation, work hours, etc.
Constraints:

Features such as categorical variables need to be encoded to make them suitable for machine learning models.
The model should balance performance and interpretability, especially when deploying it in real-world applications.
'''

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the datasets
train_data = pd.read_csv('E:/Honars(DS)/Data Science/Naive Bayes Assignment/SalaryData_Train.csv')
test_data = pd.read_csv('E:/Honars(DS)/Data Science/Naive Bayes Assignment/SalaryData_Test.csv')

# 3. Data Pre-processing
def preprocess_data(data):
    # Handle missing values
    data = data.dropna()

    # Drop redundant columns
    if 'education-num' in data.columns:
        data = data.drop('education-num', axis=1)
    
    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    return data, label_encoders

# Preprocess train and test datasets
train_data, train_encoders = preprocess_data(train_data)
test_data, _ = preprocess_data(test_data)

# Separate features and target variable
X_train = train_data.drop('Salary', axis=1)
y_train = train_data['Salary']
X_test = test_data.drop('Salary', axis=1)
y_test = test_data['Salary']

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Exploratory Data Analysis (EDA)
# Summary Statistics
print("Summary of Train Dataset:")
print(train_data.describe())

# Univariate Analysis: Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(train_data['age'], kde=True, bins=20, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis: Education vs Salary
plt.figure(figsize=(10, 6))
sns.countplot(x='education', hue='Salary', data=train_data)
plt.title('Education vs Salary')
plt.xticks(rotation=45)
plt.xlabel('Education')
plt.ylabel('Count')
plt.legend(title='Salary', labels=['<=50K', '>50K'])
plt.show()

# 5. Model Building
# Train a Gaussian Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 6. Benefits/Impact of the Solution
print("""
Benefits/Impact:
- Helps predict individuals' income categories, enabling targeted policies and decisions.
- Useful for workforce analysis and job market studies.
- Enables better segmentation for business planning and resource allocation.
""")
