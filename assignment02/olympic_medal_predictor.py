#### ASSIGNMENT 01 - AIDI2000 ####

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Load the dataset
data = pd.read_excel('medals-dataset/Olympic-Medals.xlsx')

# Create a directory for saving visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Configure pandas to display float numbers with 4 decimal places
pd.set_option('display.float_format', '{:.4f}'.format)

# Display the first few rows of the dataset
print(data.head())

# Check for missing/null values
print(data.isnull().sum())

# Check Basic Descriptive statistics
print(data.describe())

# Analyze categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical columns:", list(categorical_cols))

# Analyze numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Numerical columns:", list(numerical_cols))