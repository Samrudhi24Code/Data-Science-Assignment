# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:14:32 2025

@author: Dell
"""

''' Problem statement
A glass manufacturing plant uses different earth 
elements to design new glass materials based on customer
requirements. For that, they would like to automate the 
process of classification as it’s a tedious job to manually 
classify them. Help the company achieve its objective by correctly
classifying the glass type based on the other features 
using KNN algorithm.'''

### 1. Business Problem

# 1.1. What is the business objective?
# - Create an automated system to classify glass types, making the manufacturing process faster and more efficient.
# - Improve the accuracy of glass classification to meet customer needs more precisely.
# - Reduce the amount of manual work and save time by using automation for classification.

# 1.2. Are there any constraints?
# - The model must be very accurate because wrong classifications could lead to poor-quality glass.
# - The availability of data might limit the model’s performance, especially if some glass types are underrepresented.
# - There may be limited resources to handle the extra computational costs if the model is used in a real-time production setting.

# Import required libraries
import pandas as pd  # To handle data in DataFrame format
import numpy as np  # For numerical operations, e.g., creating arrays
import matplotlib.pyplot as plt  # To plot graphs for visualization
import seaborn as sns  # For more advanced visualization (e.g., boxplots)
from sklearn.preprocessing import StandardScaler  # To standardize the features
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier for classification
from sklearn.metrics import accuracy_score  # To evaluate the model's performance

# Load the dataset from a specified path
data = pd.read_csv("E:\\Honars(DS)\\Data Science\\K-Nearest Assignment\\glass.csv")
print(data.columns)  # Display column names to understand the structure of the dataset

# Perform some initial exploration of the dataset
data.head()  # View the first few rows of the dataset to understand the features
data.isnull().sum()  # Check for missing values in the dataset
data.describe()  # Get a statistical summary of the dataset, including mean, std deviation, etc.

# Data Preprocessing
features = data.drop(columns=["Type"])  # Drop the target column "Type" to keep only features
target = data['Type']  # Extract the target column "Type" (glass type)

# Standardize the features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # Scale the features to have a mean of 0 and standard deviation of 1
data_scaled = pd.DataFrame(features_scaled, columns=features.columns)  # Convert the scaled data back to a DataFrame
data_scaled['Type'] = target  # Add the target column back to the scaled data to keep track of the labels
data_scaled.head()  # View the scaled data

# Plot boxplots to visualize outliers in the data
sns.boxplot(data=data_scaled)  # A boxplot helps in identifying outliers in numerical data

# Function to cap outliers using the IQR (Interquartile Range) method
def cap_outliers(df, columns):
    for col in columns:
        # Calculate the first (Q1) and third quartiles (Q3) for each column
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        # Determine the lower and upper bounds for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Cap values outside of the bounds to the respective bounds
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

# Apply the outlier capping function to the scaled dataset
data_capped = cap_outliers(data_scaled, data_scaled.columns)

# Standardize the features again after capping the outliers
features = data_capped.drop(columns=["Type"])  # Drop the target column again
target = data_capped['Type']  # Extract the target column
features_scaled = scaler.fit_transform(features)  # Standardize the features again
data_scaled = pd.DataFrame(features_scaled, columns=features.columns)  # Convert back to a DataFrame
data_scaled['Type'] = target  # Add the target column back
data_scaled.head()  # View the updated data

# Plot boxplots again to check for outliers after capping them
sns.boxplot(data=data_scaled)

# Split the data into features (X) and target (y)
X = data_scaled.drop('Type', axis=1)  # Features (all columns except the target)
y = data_scaled['Type']  # Target (glass type)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors for the KNN algorithm
knn.fit(X_train, y_train)  # Train the model on the training data

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy using accuracy_score
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
print(f"Accuracy: {accuracy}")  # Print the accuracy of the model

# Find the best value for k (number of neighbors) by testing different values
acc = []  # List to store accuracy values for different k values
for i in range(3, 50, 2):  # Test odd values of k from 3 to 49
    knn = KNeighborsClassifier(n_neighbors=i)  # Initialize KNN with a specific k value
    knn.fit(X_train, y_train)  # Train the model with this k value
    y_pred = knn.predict(X_test)  # Predict on the test data
    # Calculate training and testing accuracy for this k value
    train_acc = np.mean(knn.predict(X_train) == y_train)  # Accuracy on the training set
    test_acc = np.mean(y_pred == y_test)  # Accuracy on the testing set
    acc.append([train_acc, test_acc])  # Store both accuracies for analysis

# Plot training and testing accuracy for different k values
plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], 'ro-', label="Training Accuracy")  # Red line for training accuracy
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], 'bo-', label="Testing Accuracy")  # Blue line for testing accuracy
plt.xlabel('Number of Neighbors (k)')  # Label for x-axis (k value)
plt.ylabel('Accuracy')  # Label for y-axis (accuracy)
plt.legend()  # Show legend for the plot
plt.show()  # Display the plot

# After checking for the best k, try specific values of k based on results from the plot
# Try k=7 for classification
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)  # Train the model with k=7
y_pred = knn.predict(X_test)  # Make predictions on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy
print(f"Accuracy with k=7: {accuracy}")  # Print the accuracy for k=7

# Try k=9 for classification
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)  # Train the model with k=9
y_pred = knn.predict(X_test)  # Make predictions on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy
print(f"Accuracy with k=9: {accuracy}")  # Print the accuracy for k=9

# Try k=43 for classification
knn = KNeighborsClassifier(n_neighbors=43)
knn.fit(X_train, y_train)  # Train the model with k=43
y_pred = knn.predict(X_test)  # Make predictions on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy
print(f"Accuracy with k=43: {accuracy}")  # Print the accuracy for k=43
