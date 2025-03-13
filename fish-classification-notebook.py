# Fish Market Dataset - Classification Model

## Step 1: Load and Explore Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Fish.csv")

# Display first few rows
print("First 5 rows of the dataset:")
df.head()

## Step 2: Check data information

# Check data types and missing values
print("\nDataset Info:")
df.info()

print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
df.describe()

## Step 3: Exploratory Data Analysis (EDA)

# Check distribution of fish species
plt.figure(figsize=(10, 6))
sns.countplot(x='Species', data=df)
plt.xticks(rotation=45)
plt.title("Distribution of Fish Species")
plt.tight_layout()
plt.show()

# Visualize the relationship between features
plt.figure(figsize=(12, 8))
sns.pairplot(df, hue="Species", height=2)
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation = df.iloc[:, 1:].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.show()

# Box plots for numerical features by species
plt.figure(figsize=(15, 10))
for i, feature in enumerate(['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='Species', y=feature, data=df)
    plt.xticks(rotation=45)
    plt.title(f'Box Plot of {feature} by Species')
plt.tight_layout()
plt.show()

## Step 4: Data Preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encode the target variable (Species)
encoder = LabelEncoder()
df['Species_encoded'] = encoder.fit_transform(df['Species'])

# Map numerical labels to original species
species_mapping = {i: species for i, species in enumerate(encoder.classes_)}
print("\nSpecies mapping:")
for key, value in species_mapping.items():
    print(f"{key}: {value}")

# Define features (X) and target (y) for classification
X = df.drop(columns=["Species", "Species_encoded"])  
y = df["Species_encoded"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

## Step 5: Build and Evaluate Multiple Models

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

# Compare model performances
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Step 6: Hyperparameter Tuning for the Best Model

from sklearn.model_selection import GridSearchCV

# Based on previous results, let's assume Random Forest performed best
# Hyperparameter tuning for Random Forest 
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Train final model with best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_scaled, y_train)

# Evaluate final model
y_pred = best_rf.predict(X_test_scaled)
print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
plt.figure(figsize=(10, 6))
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

## Step 7: Save the Model and Preprocessing Objects

import joblib

# Save the model
joblib.dump(best_rf, "fish_species_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Save the encoder with class mappings
joblib.dump(encoder, "encoder.pkl")

print("\nModel and preprocessing objects saved successfully.")

## Step 8: Test Model Loading and Prediction

# Load the model
loaded_model = joblib.load("fish_species_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
loaded_encoder = joblib.load("encoder.pkl")

# Create a sample input
sample = X_test.iloc[0].values.reshape(1, -1)
sample_scaled = loaded_scaler.transform(sample)

# Make prediction
prediction = loaded_model.predict(sample_scaled)
predicted_species = loaded_encoder.inverse_transform(prediction)[0]

print("\nSample Input:")
print(X_test.iloc[0])
print("\nPredicted Species:", predicted_species)

# Function for making predictions (to be used in Flask application)
def predict_species(weight, length1, length2, length3, height, width):
    # Create input array
    input_data = np.array([[weight, length1, length2, length3, height, width]])
    
    # Scale input
    input_scaled = loaded_scaler.transform(input_data)
    
    # Predict species index
    prediction = loaded_model.predict(input_scaled)
    
    # Convert index to species name
    predicted_species = loaded_encoder.inverse_transform(prediction)[0]
    
    return predicted_species

# Test function with sample data
test_result = predict_species(
    weight=X_test.iloc[0]['Weight'],
    length1=X_test.iloc[0]['Length1'],
    length2=X_test.iloc[0]['Length2'],
    length3=X_test.iloc[0]['Length3'],
    height=X_test.iloc[0]['Height'],
    width=X_test.iloc[0]['Width']
)

print("\nTest Prediction Function Result:", test_result)
