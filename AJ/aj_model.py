from gettext import install

import pandas as pd
import pip
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
try:
    import dagshub
except ModuleNotFoundError:
    dagshub = None
import mlflow
import requests
import subprocess, sys, importlib
import os
import shutil
import pickle

# Load the data
data = pd.read_csv(r'C:\Users\dell\Desktop\Demo dvc\skillfy_190725\data\winequality-red.csv')

# Assume 'quality' is the target column
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
if dagshub is not None:
    dagshub.init(repo_owner='Anuragas0326835', repo_name='skillfy_190725', mlflow=True)
else:
    print("dagshub not installed; proceeding without Dagshub integration.")
with mlflow.start_run():
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score_weighted", f1)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    
    mlflow.sklearn.log_model(clf, "random_forest_model")    

    # Save the model as a pickle file
    model_path = "random_forest_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Log the pickle file as an artifact in MLflow (and thus Dagshub)
    mlflow.log_artifact(model_path)

    # Optionally, remove the pickle file after logging
    os.remove(model_path)