#### ASSIGNMENT 01 - AIDI2000 ####

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_excel('medals-dataset/Olympic-Medals.xlsx')

# Create a directory for saving visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Configure pandas to display float numbers with 4 decimal places
#pd.set_option('display.float_format', '{:.4f}'.format)

# Display the first few rows of the dataset
#print(data.head())

# Check for missing/null values
#print(data.isnull().sum())

# Check Basic Descriptive statistics
#print(data.describe())

# Drop unnecessary/redundant columns
data = data.drop(columns=['Country-Code','Gender-Code'])

# Analyze categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical columns:", list(categorical_cols))

# Analyze numerical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("Numerical columns:", list(numerical_cols))

# EDA Visualization 01 - Distribution of Medals
medal_colors = {
    'Gold': '#FFD700',    # Gold
    'Silver': '#C0C0C0',  # Silver
    'Bronze': '#CD7F32'   # Bronze
}

""" plt.figure(figsize=(10, 6))
sns.countplot(x='Medal', data=data, palette=medal_colors, hue='Medal')
plt.title('Distribution of Medal Types', fontsize=16)
plt.xlabel('Medal Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('visualizations/01-medal_distribution.png')
plt.close() """

# EDA Visualization 02 - Medal Count by Country (Top 15)
""" plt.figure(figsize=(10, 6))
country_medals = data['Country'].value_counts().head(15)
sns.barplot(x=country_medals.index, y=country_medals.values, palette='viridis', hue=country_medals.index)
plt.title('Top 15 Countries by Medal Count', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Medals', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/02-top_15_countries_medals.png')
plt.close() """

# EDA Visualization 03 - Medal Count by Country (Top 10) breakdown by Medal Type
""" plt.figure(figsize=(10, 6))
medals_by_country = data.groupby(['Country', 'Medal']).size().unstack(fill_value=0)
medals_by_country['Total'] = medals_by_country.sum(axis=1)
top10_countries = medals_by_country.sort_values('Total', ascending=False).head(10)
top10_countries = top10_countries.drop(columns='Total')
top10_countries[['Gold', 'Silver', 'Bronze']].plot(kind='bar', stacked=False, figsize=(12, 6), color=['#FFD700', '#C0C0C0', '#CD7F32'])
plt.title('Top 10 Countries by Medal Count and Type', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Medals', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Medal Type')
plt.tight_layout()
plt.savefig('visualizations/03-top_10_countries_breakdown_medals.png')
plt.close() """

# EDA Visualization 04 - Medal Count by Sport and Gender
""" plt.figure(figsize=(10, 6))
medals_by_sport_gender = data.groupby(['Sport', 'Gender']).size().unstack(fill_value=0)
medals_by_sport_gender.plot(kind='bar', stacked=True, figsize=(12, 6), color=['blue', 'magenta'])
plt.title('Medal Count by Sport and Gender', fontsize=16)
plt.xlabel('Sport', fontsize=14)
plt.ylabel('Number of Medals', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('visualizations/04-medal_count_sport_gender.png')
plt.close() """

# EDA Visualization 05 - Medal Count by Year
""" plt.figure(figsize=(10, 6))
medals_by_year = data.groupby('Year').size()
medals_by_year.plot(kind='line', marker='o')
plt.title('Total Medal Count by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Medals', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/05-medal_count_by_year.png')
plt.close() """

# Create a copy of the dataframe for preprocessing
data_encoded = data.copy()

# Encode the target variable (Medal)
label_encoder = LabelEncoder()
data_encoded['Medal_Encoded'] = label_encoder.fit_transform(data_encoded['Medal'])
print(f"Medal encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Encode other categorical variables
#print("\nEncoding other categorical variables...")
for col in categorical_cols:
    if col != 'Medal':  # Medal is already encoded
        data_encoded[f'{col}_Encoded'] = label_encoder.fit_transform(data_encoded[col])
        print(f"Encoded {col} -> {col}_Encoded")

# One-hot encoding for categorical variables
print("\nPerforming one-hot encoding for categorical variables...")
df_onehot = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'Medal'])
print(f"Shape after one-hot encoding: {df_onehot.shape}")