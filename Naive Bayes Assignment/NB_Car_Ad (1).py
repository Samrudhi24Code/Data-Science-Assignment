# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:15:12 2025

@author: Dell

"""

'''Problem Statement: -
This dataset contains information of users in a social network. 
This social network has several business clients which can post 
ads on it. One of the clients has a car company which has just 
launched a luxury SUV for a ridiculous price. Build a Bernoulli 
Naïve Bayes model using this dataset and classify which of the users 
of the social network are going to purchase this luxury SUV. 
1 implies that there was a purchase and 0 implies there wasn’t a purchase.

'''

# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# Load the dataset
file_path = 'E:/Honars(DS)/Data Science/Naive Bayes Assignment/NB_Car_Ad.csv'  
data = pd.read_csv(file_path)

# Constraints
data_constraints = """
Constraints:
1. Data Constraints:
   - The dataset assumes that all numerical features are scaled.
   - Categorical features like 'Gender' must be encoded before model building.
   - There must be no missing or null values in the dataset.
   - The dataset should not contain duplicate entries.
   
2. Business Constraints:
   - The car company needs quick predictions for targeted advertisements.
   - The model must be interpretable to allow business stakeholders to understand the predictions.
   - Marketing budgets are limited, so high precision in identifying likely buyers is critical to avoid wasted resources.
   - Data privacy must be maintained as this involves sensitive user information.
   - The company may need periodic model retraining as new data (or users) enters the system.
"""
print("Constraints:")
print(data_constraints)

# Data Dictionary
data_description = {
    "Feature": ["User ID", "Gender", "Age", "EstimatedSalary", "Purchased"],
    "Data Type": ["int64", "object", "int64", "int64", "int64"],
    "Relevance": [
        "Irrelevant, as it is a unique identifier.",
        "Relevant, can help identify purchasing behavior differences between genders.",
        "Relevant, age might influence purchasing decisions.",
        "Relevant, salary could indicate purchasing power.",
        "Target variable for classification."
    ],
    "Description": [
        "Unique ID assigned to each user.",
        "Indicates the gender of the user (Male/Female).",
        "Age of the user in years.",
        "Estimated annual income of the user.",
        "Indicates if the user purchased the SUV (1 = Yes, 0 = No)."
    ]
}
data_dict = pd.DataFrame(data_description)
print("\nData Dictionary:")
print(data_dict)

# Data Preprocessing
data_cleaned = data.drop(columns=["User ID"])  # Remove irrelevant columns

# Encode the Gender column
label_encoder = LabelEncoder()
data_cleaned["Gender"] = label_encoder.fit_transform(data_cleaned["Gender"])

# Scale numerical features
scaler = StandardScaler()
data_cleaned[["Age", "EstimatedSalary"]] = scaler.fit_transform(data_cleaned[["Age", "EstimatedSalary"]])

# EDA: Summary statistics
print("\nSummary Statistics:")
print(data_cleaned.describe())

# Univariate Analysis
plt.figure(figsize=(12, 5))
sns.countplot(data=data_cleaned, x='Purchased')
plt.title("Distribution of Target Variable (Purchased)")
plt.show()

# Bivariate Analysis
plt.figure(figsize=(12, 5))
sns.boxplot(data=data_cleaned, x='Purchased', y='Age')
plt.title("Age Distribution by Purchase")
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(data=data_cleaned, x='Purchased', y='EstimatedSalary')
plt.title("Estimated Salary Distribution by Purchase")
plt.show()

# Splitting the dataset
X = data_cleaned.drop(columns=["Purchased"])
y = data_cleaned["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Building
bnb_model = BernoulliNB()
bnb_model.fit(X_train, y_train)

# Predictions
y_pred = bnb_model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Business Impact
business_impact = """
The solution helps the car company identify potential buyers of the luxury SUV, 
enabling targeted marketing strategies. By focusing on the users most likely to 
purchase the SUV, the company can optimize advertising budgets and maximize sales 
conversions. This predictive model reduces guesswork and allows data-driven decision-making.
"""
print("\nBusiness Impact:")
print(business_impact)