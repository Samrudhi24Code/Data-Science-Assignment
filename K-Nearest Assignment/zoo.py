# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:14:32 2025

@author: Dell
"""

''' Problem statement
A national zoo park in India is dealing with the problem of 
segregating animals based on the different attributes they have 
(e.g., weight, height, diet, habitat). The zoo wants to automate 
the process of classifying animals using machine learning techniques 
to improve their management and resource allocation. 
Help the zoo by building a KNN model that can classify animals based 
on their features.'''

### 1. Business Problem

# 1.1. What is the business objective?
# - Automate the classification of animals to ensure proper segregation.
# - The zoo aims to categorize animals based on attributes (e.g., weight, diet, size, habitat).
# - Improve animal care by grouping animals with similar requirements (e.g., same habitat or food).

# 1.2. Are there any constraints?
# - The classification must be accurate to ensure animals are placed in the right groups.
# - Lack of sufficient data for some animal types may impact the model's accuracy.
# - The system needs to be efficient to handle real-time animal classification for incoming zoo visitors.

# Importing required libraries
import pandas as pd  # For handling data in DataFrame format
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization of results
import seaborn as sns  # For advanced plotting (e.g., heatmaps, pairplots)
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier to train the model
from sklearn.metrics import accuracy_score  # For evaluating model performance

# Load the dataset containing animal attributes (for illustration, assuming a CSV file)
data = pd.read_csv("E:\\AnimalDataset\\zoo_data.csv")  # Update the file path according to your dataset location

# Display the column names to understand the structure of the dataset
print(data.columns)  # Show the column names of the dataset

# Initial exploration of the dataset
data.head()  # Display the first few rows of the dataset to understand the features
data.isnull().sum()  # Check for missing values in the dataset
data.describe()  # Get statistical summary to understand distribution, mean, etc.

# Data Preprocessing
# Drop the target column 'Class' which indicates the animal class (output)
features = data.drop(columns=["Class"])  # Features (attributes of animals)
target = data["Class"]  # Target column representing the animal classification

# Scaling the features using StandardScaler to standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # Standardize the features to mean=0, std=1
data_scaled = pd.DataFrame(features_scaled, columns=features.columns)  # Convert back to DataFrame
data_scaled['Class'] = target  # Add the target column back to the DataFrame to keep track of the labels
data_scaled.head()  # View the scaled data

# Visualizing the dataset using a pairplot to see the relationship between features
sns.pairplot(data_scaled, hue='Class')  # Color by 'Class' (animal type)

# Splitting the data into features (X) and target (y)
X = data_scaled.drop('Class', axis=1)  # Feature set (all columns except the target 'Class')
y = data_scaled['Class']  # Target labels (animal classes)

# Splitting the dataset into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors for classification
knn.fit(X_train, y_train)  # Train the KNN model on the training data

# Predict the animal classes using the trained model on the test data
y_pred = knn.predict(X_test)  # Predict class labels on the test set

# Evaluate the accuracy of the model by comparing predictions with actual values
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy score of the model
print(f"Accuracy of the KNN model: {accuracy * 100:.2f}%")  # Print the accuracy in percentage

# Find the optimal value for k (number of neighbors) by testing various values of k
acc = []  # List to store accuracy values for different k values
for i in range(3, 50, 2):  # Test odd values of k from 3 to 49
    knn = KNeighborsClassifier(n_neighbors=i)  # Initialize KNN classifier with the current k
    knn.fit(X_train, y_train)  # Train the model
    y_pred = knn.predict(X_test)  # Make predictions on the test data
    # Calculate the accuracy for this k value
    test_acc = np.mean(y_pred == y_test)  # Test accuracy
    acc.append(test_acc)  # Store the test accuracy for this k

# Plot the accuracy for different values of k
plt.plot(np.arange(3, 50, 2), acc, 'bo-', label="Testing Accuracy")  # Plot testing accuracy
plt.xlabel('Number of Neighbors (k)')  # Label for x-axis
plt.ylabel('Accuracy')  # Label for y-axis
plt.title('KNN Classifier Accuracy vs. Number of Neighbors')  # Title of the plot
plt.legend()  # Show the legend on the plot
plt.show()  # Display the plot

# Try the model with different k values based on the plot (e.g., k=7, k=9, k=11)
knn = KNeighborsClassifier(n_neighbors=7)  # Example with k=7
knn.fit(X_train, y_train)  # Train the model
y_pred = knn.predict(X_test)  # Make predictions
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy with k=7: {accuracy * 100:.2f}%")  # Print the accuracy for k=7

knn = KNeighborsClassifier(n_neighbors=9)  # Example with k=9
knn.fit(X_train, y_train)  # Train the model
y_pred = knn.predict(X_test)  # Make predictions
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy with k=9: {accuracy * 100:.2f}%")  # Print the accuracy for k=9

knn = KNeighborsClassifier(n_neighbors=11)  # Example with k=11
knn.fit(X_train, y_train)  # Train the model
y_pred = knn.predict(X_test)  # Make predictions
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy with k=11: {accuracy * 100:.2f}%")  # Print the accuracy for k=11

# Final inference and recommendations:
# - Based on the accuracy scores and the plot of accuracy vs. k, we can choose the optimal value for k that 
#   results in the best balance of accuracy and computational efficiency.
# - The KNN model seems to perform well in classifying animals based on their attributes.
# - It is important to ensure that the dataset is large and well-balanced to avoid any biases in classification.
# - The zoo can use this model to automatically classify new animals and group them according to their attributes 
#   to optimize management and care within the zoo.
