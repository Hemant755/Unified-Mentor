# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle  # For model deployment

# Step 2: Load the dataset
# Replace 'healthcare_data.csv' with your dataset
data = pd.read_csv(r'C:\Users\theba\OneDrive\Desktop\Projects\Project 3 Analyzing Personalize Healthcare Recommandation\blood.csv')

# Step 3: Data exploration and visualization
# Basic information and statistics
print("Dataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Visualize relationships and distributions
sns.pairplot(data, diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Data Preprocessing
# Separate features and target variable
X = data.drop('Class', axis=1)  # Target column is 'Class'
y = data['Class']

# Identify numerical and categorical features
numerical_features = ['Recency', 'Frequency', 'Monetary', 'Time']  # Adjust as necessary
categorical_features = []  # If no categorical features, leave as empty list

# Create pipelines for preprocessing
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

if categorical_features:
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
else:
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features)
    ])

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection and Training
# Create a model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the trained model for deployment
with open('healthcare_model.pkl', 'wb') as file:
    pickle.dump(model_pipeline, file)

# Step 7: Model Evaluation
# Make predictions
y_pred = model_pipeline.predict(X_test)

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Curve
if len(np.unique(y)) == 2:  # Binary classification check
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=model_pipeline.named_steps['classifier'].classes_[1])
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Step 8: Recommendation System Implementation
def generate_recommendations(patient_data):
    # Predict the category
    prediction = model_pipeline.predict(patient_data)
    recommendation_mapping = {
        0: 'No action needed',
        1: 'Regular check-up required'
    }
    return recommendation_mapping.get(prediction[0], "Recommendation unavailable")

# Example: Generate recommendations for a new patient
example_patient_data = pd.DataFrame({
    'Recency': [10],
    'Frequency': [5],
    'Monetary': [1500],
    'Time': [30]
})

print("\nPersonalized Recommendation:")
print(generate_recommendations(example_patient_data))

# Step 9: Deployment
# Use Flask for deployment (code snippet for Flask app)
'''
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model
with open('healthcare_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data
    data = request.get_json()
    patient_data = pd.DataFrame(data)
    recommendation = generate_recommendations(patient_data)
    return jsonify({'recommendation': recommendation})

if __name__ == '__main__':
    app.run(debug=True)
'''

# Step 10: Documentation
# Add comments throughout the code
# Use Jupyter Markdown cells if working in a notebook
# Create structured documentation for steps and results


