#### ASSIGNMENT 01 - AIDI2000 ####

# Import necessary libraries
import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf  
from tensorflow import keras  
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance

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

plt.figure(figsize=(10, 6))
sns.countplot(x='Medal', data=data, palette=medal_colors, hue='Medal')
plt.title('Distribution of Medal Types', fontsize=16)
plt.xlabel('Medal Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('visualizations/01-medal_distribution.png')
plt.close()

# EDA Visualization 02 - Medal Count by Country (Top 15)
plt.figure(figsize=(10, 6))
country_medals = data['Country'].value_counts().head(15)
sns.barplot(x=country_medals.index, y=country_medals.values, palette='viridis', hue=country_medals.index)
plt.title('Top 15 Countries by Medal Count', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Medals', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/02-top_15_countries_medals.png')
plt.close()

# EDA Visualization 03 - Medal Count by Country (Top 10) breakdown by Medal Type
plt.figure(figsize=(10, 6))
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
plt.close()

# EDA Visualization 04 - Medal Count by Sport and Gender
plt.figure(figsize=(10, 6))
medals_by_sport_gender = data.groupby(['Sport', 'Gender']).size().unstack(fill_value=0)
medals_by_sport_gender.plot(kind='bar', stacked=True, figsize=(12, 6), color=['blue', 'magenta'])
plt.title('Medal Count by Sport and Gender', fontsize=16)
plt.xlabel('Sport', fontsize=14)
plt.ylabel('Number of Medals', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('visualizations/04-medal_count_sport_gender.png')
plt.close()

# EDA Visualization 05 - Medal Count by Year
plt.figure(figsize=(10, 6))
medals_by_year = data.groupby('Year').size()
medals_by_year.plot(kind='line', marker='o')
plt.title('Total Medal Count by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Medals', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/05-medal_count_by_year.png')
plt.close()

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

# Generate Logistic Regression classification report
log_report = classification_report(y_test, y_pred_log, target_names=medal_encoder.classes_)

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

# Generate Decision Tree classification report
dec_report = classification_report(y_test, y_pred_dectree, target_names=medal_encoder.classes_)  

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

# Split the data into training and validation sets
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X_normalized, y_categorical, test_size=0.2, random_state=42)

# Build the neural network model
print("\nBuilding neural network model...")
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with validation split
print("\nTraining the neural network...")
start_time = time.time()
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
training_time = time.time() - start_time

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
nnt_loss, nnt_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test accuracy: {nnt_accuracy:.4f}")

# Generate Neural Network classification report
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)
nn_report = classification_report(y_true, y_pred, target_names=medal_encoder.classes_)

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
    Dense(16, activation='relu', input_shape=(enhanced_X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
enhanced_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with validation split
print("\nTraining the neural network...")
start_time_enhanced = time.time()
enhanced_history = enhanced_model.fit(
    enhanced_X_train, enhanced_y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
training_time_enhanced = time.time() - start_time_enhanced

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
enhanced_nnt_loss, enhanced_nnt_accuracy = enhanced_model.evaluate(enhanced_X_test, enhanced_y_test_cat, verbose=0)
print(f"\nTest accuracy (enhanced set): {enhanced_nnt_accuracy:.4f}")

# Generate classification report
enhanced_y_pred_probs = enhanced_model.predict(enhanced_X_test, verbose=0)
enhanced_y_pred = np.argmax(enhanced_y_pred_probs, axis=1)
enhanced_y_true = np.argmax(enhanced_y_test_cat, axis=1)
enhanced_nn_report = classification_report(enhanced_y_true, enhanced_y_pred, target_names=medal_encoder.classes_)

# ----------------------------------------------------- #

# Decision Tree Feature Importance
feature_names = X.columns
dec_importances = decTree.feature_importances_

# Sort feature importances in descending order
sorted_indices = np.argsort(dec_importances)[::-1]
sorted_importances = dec_importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# EDA Visualization 10 - Decision Tree Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=45, ha='right')
plt.title('Features Importance List - Decision Tree', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/10-decision_tree_feature_importance.png')
plt.close()

# ----------------------------------------------------- #

# Create a wrapper class for the neural network to make it compatible with scikit-learn
class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y):
        # Already fitted, just return self
        return self
    
    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

# Create a wrapper for the neural network
nn_wrapper = NeuralNetworkWrapper(model)

# Calculate feature importance using permutation importance
nn_perm_importance = permutation_importance(nn_wrapper, X_normalized, y, n_repeats=5, random_state=42)
nn_importances = nn_perm_importance.importances_mean

# Sort feature importances
nn_sorted_idx = np.argsort(nn_importances)[::-1]
nn_sorted_importances = nn_importances[nn_sorted_idx]
nn_sorted_feature_names = np.array(feature_names)[nn_sorted_idx]

# EDA Visualization 11 - Neural Network Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(nn_sorted_importances)), nn_sorted_importances, align='center')
plt.xticks(range(len(nn_sorted_importances)), nn_sorted_feature_names, rotation=45, ha='right')
plt.title('Features Importance List - Neural Network', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/11-neural_network_feature_importance.png')
plt.close()

# Enhanced Neural Network Features
enhanced_feature_names = enhanced_X.columns

# Create a wrapper for the enhanced neural network
enhanced_nn_wrapper = NeuralNetworkWrapper(enhanced_model)

# Calculate feature importance using permutation importance
enhanced_perm_importance = permutation_importance(enhanced_nn_wrapper, enhanced_X_normalized, np.argmax(y_categorical, axis=1), n_repeats=3, random_state=42)
enhanced_importances = enhanced_perm_importance.importances_mean

# Sort feature importances
enhanced_sorted_idx = np.argsort(enhanced_importances)[::-1]
enhanced_sorted_importances = enhanced_importances[enhanced_sorted_idx]
enhanced_sorted_feature_names = np.array(enhanced_feature_names)[enhanced_sorted_idx]

# EDA Visualization 12 - Enhanced Neural Network Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(enhanced_sorted_importances)), enhanced_sorted_importances, align='center')
plt.xticks(range(len(enhanced_sorted_importances)), enhanced_sorted_feature_names, rotation=45, ha='right')
plt.title('Features Importance List - Enhanced Neural Network', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/12-enhanced_neural_network_feature_importance.png')
plt.close()

# ----------------------------------------------------- #

# Compare with previous models
print("\n--- Model Comparison (All Models) ---")
print(f"\nLogistic Regression Accuracy: {log_accuracy:.4f}")
print(f"Decision Tree Accuracy: {dec_accuracy:.4f}")
print(f"Neural Network Accuracy: {nnt_accuracy:.4f}")
print(f"Enhanced Neural Network Accuracy: {enhanced_nnt_accuracy:.4f}")

# Determine the best model among all three
accuracies = {
    "Logistic Regression": log_accuracy,
    "Decision Tree": dec_accuracy,
    "Neural Network": nnt_accuracy,
    "Enhanced Neural Network": enhanced_nnt_accuracy
}
best_model = max(accuracies, key=accuracies.get)
print(f"\nThe best performing model is: {best_model} with accuracy {accuracies[best_model]:.4f}")

# Print classification reports for all models
print("\n--- Classification Reports ---")
print("\nLogistic Regression Classification Report:")
print(log_report)
print("\nDecision Tree Classification Report:")
print(dec_report)
print("\nNeural Network Classification Report:")
print(nn_report)
print("\nEnhanced Neural Network Classification Report:")
print(enhanced_nn_report)   

# Create table headers
print(f"{'Rank':<5}{'Decision Tree':<20}{'Importance':<13}{'Neural Network':<20}{'Importance':<13}{'Enhanced Neural Net':<20}{'Importance':<10}")
print("-" * 101)

# Determine how many features to display (Top 5)
num_features = min(5, len(sorted_feature_names), len(nn_sorted_feature_names), len(enhanced_sorted_feature_names))

# Print each row of the table
for i in range(num_features):
    dt_feature = sorted_feature_names[i] if i < len(sorted_feature_names) else "-"
    dt_importance = f"{sorted_importances[i]:.4f}" if i < len(sorted_importances) else "-"
    
    nn_feature = nn_sorted_feature_names[i] if i < len(nn_sorted_feature_names) else "-"
    nn_importance = f"{nn_sorted_importances[i]:.4f}" if i < len(nn_sorted_importances) else "-"
    
    enh_feature = enhanced_sorted_feature_names[i] if i < len(enhanced_sorted_feature_names) else "-"
    enh_importance = f"{enhanced_sorted_importances[i]:.4f}" if i < len(enhanced_sorted_importances) else "-"
    
    print(f"{i+1:<5}{dt_feature:<20}{dt_importance:<13}{nn_feature:<20}{nn_importance:<13}{enh_feature:<20}{enh_importance:<10}")

# ----------------------------------------------------- #

# Summary reflection on features
print("\n--- Summary Reflection ---")
print("\n1. Did accuracy improve?")
if enhanced_nnt_accuracy > nnt_accuracy:
    print(f"   Yes, accuracy improved by {enhanced_nnt_accuracy - nnt_accuracy:.4f} ({(enhanced_nnt_accuracy - nnt_accuracy) / nnt_accuracy * 100:.2f}%)")
else:
    print(f"   No, accuracy decreased by {nnt_accuracy - enhanced_nnt_accuracy:.4f} ({(nnt_accuracy - enhanced_nnt_accuracy) / nnt_accuracy * 100:.2f}%)")

print("\n2. Did the model train faster/slower?")
if training_time_enhanced < training_time:
    print(f"   The new model was faster by {training_time - training_time_enhanced:.2f} seconds ({(training_time - training_time_enhanced) / training_time * 100:.2f}%)")
else:
    print(f"   The new model was slower by {training_time_enhanced - training_time:.2f} seconds ({(training_time_enhanced - training_time) / training_time * 100:.2f}%)")

print("\n3. Which feature seemed more predictive?")
print("   Based on the feature importance analysis:")
print(f"   - For Decision Tree: {sorted_feature_names[0]} (Importance: {sorted_importances[0]:.4f}) and {sorted_feature_names[1]} (Importance: {sorted_importances[1]:.4f})")
print(f"   - For Neural Network: {nn_sorted_feature_names[0]} (Importance: {nn_sorted_importances[0]:.4f}) and {nn_sorted_feature_names[1]} (Importance: {nn_sorted_importances[1]:.4f})")
print(f"   - For Enhanced Neural Network: {enhanced_sorted_feature_names[0]} (Importance: {enhanced_sorted_importances[0]:.4f}) and {enhanced_sorted_feature_names[1]} (Importance: {enhanced_sorted_importances[1]:.4f})\n")
