# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:16:41 2024

@author: Dell
"""

'''Objective:
    Perform Singular Value Decomposition (SVD) on a matrix and apply it to a dataset to perform dimensionality reduction for clustering analysis.
    The goal is to extract the most important features (principal components) from the dataset and visualize the result using a scatter plot.
'''

##############################################################

'''
Task:
    1. Perform Singular Value Decomposition (SVD) on a sample matrix.
    2. Apply SVD to a real-world dataset, and reduce the dimensionality to three principal components.
    3. Visualize the reduced data in a scatter plot.
'''

# Problem Statement:
# The objective of this code is to use Singular Value Decomposition (SVD) to reduce the dimensionality of a dataset, enabling better clustering and analysis.
# By applying SVD, we break the data into principal components that capture the most important variance in the data.
# This can be useful for tasks such as clustering analysis, where reducing dimensionality helps uncover patterns and relationships that may be hidden in the original dataset.

# Business Objective:
# Reducing dimensionality with SVD allows businesses to simplify large datasets, making them easier to process and interpret.
# This is essential in clustering and pattern recognition tasks, where identifying trends and structures in the data can lead to valuable insights.
# By using SVD for dimensionality reduction, businesses can speed up computational processes, improve data quality, and gain a deeper understanding of the underlying structures.

# Solution:
# The solution involves performing Singular Value Decomposition (SVD) on both a sample matrix and a real-world dataset to reduce its dimensionality.
# The principal components are extracted, and then a scatter plot is generated to visualize the reduced data.
# The resulting scatter plot helps in visualizing the clustering tendencies in the data, based on the most significant components derived from SVD.

##############################################################

import numpy as np
from numpy import array
from scipy.linalg import svd 

# Sample matrix A (for demonstration of SVD)
A=array([[1,0,0,0,2],
         [0,0,3,0,0],
         [0,0,0,0,0],
         [0,4,0,0,0]])

# Display the sample matrix A
print(A)

# Perform Singular Value Decomposition (SVD)
U, d, Vt = svd(A)

# Display the U, d (singular values), and Vt matrices
print(U)
print(d)
print(Vt)

# Convert singular values 'd' to a diagonal matrix
print(np.diag(d))

#########################################

# Apply SVD to a dataset

import pandas as pd

# Load the dataset from an Excel file
data = pd.read_excel("C:/5_clustering/University_Clustering.xlsx")

# Preview the first few rows of the data
data.head()

# Remove non-numeric data columns (if any) and keep the relevant numeric columns for SVD
data = data.iloc[:, 2:]  # Assuming the first two columns are non-numeric

# Display the cleaned data
data

#########################################

from sklearn.decomposition import TruncatedSVD

# Initialize TruncatedSVD to reduce the dimensionality to 3 components
svd = TruncatedSVD(n_components=3)

# Fit the SVD model to the data
svd.fit(data)

# Transform the data using the fitted SVD model
result = pd.DataFrame(svd.transform(data))

# Display the first few rows of the transformed data
result.head()

# Name the columns as the principal components (pc0, pc1, pc2)
result.columns = ["pc0", "pc1", "pc2"]
result.head()

#########################################

# Generate a scatter plot to visualize the data based on the first two principal components (pc0 and pc1)

import matplotlib.pylab as plt

# Create a scatter plot of the data points based on pc0 and pc1
plt.scatter(x=result.pc0, y=result.pc1)

# Show the plot
plt.show()

# The scatter plot helps in visualizing the reduced data based on the most significant components.
