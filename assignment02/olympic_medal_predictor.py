#### ASSIGNMENT 01 - AIDI2000 ####

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf  # Core deep learning framework
from tensorflow import keras  # High-level API for building neural networks
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
data_features = data.drop(columns=['Medal'])  # Features
data_target = data[['Medal']]  # Target variable

# Encode the 'Medal' column
medal_encoder = LabelEncoder()
data_target = medal_encoder.fit_transform(data_target['Medal'])

# Encode other categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    if col != 'Medal': 
        data_features[f'{col}_Encoded'] = label_encoder.fit_transform(data_features[col])

# Select features and target variables
X = data_features.select_dtypes(include=['int64', 'float64'])  # Select only numeric features
y = data_target  # Use the encoded medal column as target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------------------------------- #

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions with Logistic Regression
y_pred_log = log_reg.predict(X_test)

# Evaluate Logistic Regression model
print("\nLogistic Regression Model Evaluation:")
log_accuracy = accuracy_score(y_test, y_pred_log)
print(f"Accuracy: {log_accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=medal_encoder.classes_))

# EDA Visualization 06 - Logistic Regression Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues',
           xticklabels=medal_encoder.classes_, yticklabels=medal_encoder.classes_)
plt.title('Confusion Matrix - Logistic Regression', fontsize=16)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/06-logistic_regression_confusion_matrix.png')
plt.close()

# ----------------------------------------------------- #

# Train Decision Tree model
print("\nTraining Decision Tree model...")
decTree = DecisionTreeClassifier(random_state=42)
decTree.fit(X_train, y_train)

# Make predictions with Decision Tree
y_pred_dectree = decTree.predict(X_test)

# Evaluate Decision Tree model
print("\nDecision Tree Model Evaluation:")
dec_accuracy = accuracy_score(y_test, y_pred_dectree)
print(f"Accuracy: {dec_accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dectree, target_names=medal_encoder.classes_))

# EDA Visualization 07 - Logistic Regression Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_dectree), annot=True, fmt='d', cmap='Blues',
           xticklabels=medal_encoder.classes_, yticklabels=medal_encoder.classes_)
plt.title('Confusion Matrix - Decision Tree', fontsize=16)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/07-decision_tree_confusion_matrix.png')
plt.close()

# ----------------------------------------------------- #

# Normalize numerical features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Convert target to categorical (one-hot encoding)
y_categorical = to_categorical(y)
print(f"Shape of categorical target: {y_categorical.shape}")

# Split the data into training and validation sets
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X_normalized, y_categorical, test_size=0.2, random_state=42)

# Build the neural network model
print("\nBuilding neural network model...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with validation split
print("\nTraining the neural network...")
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# EDA Visualization 08 - Neural Network Training - Validation Accuracy & Loss
plt.figure(figsize=(10, 6))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend()
# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss', fontsize=16)
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/08-neural_network_training_history.png')
plt.close()

# Evaluate the model on the test set
print("\nEvaluating neural network on test set...")
nnt_loss, nnt_accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {nnt_accuracy:.4f}")

# ----------------------------------------------------- #

# Compare with previous models
print("\n--- Model Comparison (All Models) ---")
print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")
print(f"Decision Tree Accuracy: {dec_accuracy:.4f}")
print(f"Neural Network Accuracy: {nnt_accuracy:.4f}")

# Determine the best model among all three
accuracies = {
    "Logistic Regression": log_accuracy,
    "Decision Tree": dec_accuracy,
    "Neural Network": nnt_accuracy
}
best_model = max(accuracies, key=accuracies.get)
print(f"\nThe best performing model is: {best_model} with accuracy {accuracies[best_model]:.4f}")

# ----------------------------------------------------- #

# Copying the original features after encoding for enhanced feature engineering
enhanced_data_features = data_features.copy()

# Feature 1: 'country_medal_count' - Total medals won by each country
country_medal_counts = data.groupby('Country').size()
enhanced_data_features['country_medal_count'] = enhanced_data_features['Country'].map(country_medal_counts)

# Feature 2: 'sport_popularity' - Total medals awarded in each sport
sport_popularity = data.groupby('Sport').size()
enhanced_data_features['sport_popularity'] = enhanced_data_features['Sport'].map(sport_popularity)

# Select features
enhanced_X = enhanced_data_features.select_dtypes(include=['int64', 'float64'])

# Normalize numerical features
scaler = MinMaxScaler()
enhanced_X_normalized = scaler.fit_transform(enhanced_X)

# Split the data into training and validation sets
enhanced_X_train, enhanced_X_test, enhanced_y_train_cat, enhanced_y_test_cat = train_test_split(enhanced_X_normalized, y_categorical, test_size=0.2, random_state=42)

# Build the neural network model
print("\nBuilding neural network model...")
enhanced_model = Sequential([
    Dense(64, activation='relu', input_shape=(enhanced_X_train.shape[1],)),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
enhanced_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with validation split
print("\nTraining the neural network...")
enhanced_history = enhanced_model.fit(
    enhanced_X_train, enhanced_y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# EDA Visualization 09 - Neural Network Training - Validation Accuracy & Loss Enhanced Set
plt.figure(figsize=(10, 6))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(enhanced_history.history['accuracy'], label='Training Accuracy')
plt.plot(enhanced_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend()
# Plot loss
plt.subplot(1, 2, 2)
plt.plot(enhanced_history.history['loss'], label='Training Loss')
plt.plot(enhanced_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss', fontsize=16)
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/09-neural_network_enhanced_set_training_history.png')
plt.close()

# Evaluate the model on the test set
print("\nEvaluating neural network on enhanced test set...")
enhanced_nnt_loss, enhanced_nnt_accuracy = enhanced_model.evaluate(enhanced_X_test, enhanced_y_test_cat)
print(f"Test accuracy (enhanced set): {enhanced_nnt_accuracy:.4f}")

# ----------------------------------------------------- #
