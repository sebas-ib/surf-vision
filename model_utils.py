
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate model accuracy
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for actual, pred in zip(y_test, y_pred):
        if actual == 1 and pred == 1:
            TP += 1

        elif actual == 0 and pred == 1:
            FP += 1

        elif actual == 0 and pred == 0:
            TN += 1

        elif actual == 1 and pred == 0:
            FN += 1


    accuracy = (TP + TN) / len(y_test)
    print(f"\nModel Accuracy: {accuracy:.2f}")


    # Print classification report (with sklearn.metrics)
    report = classification_report(y_test, y_pred, target_names=['Not Worth Surfing', 'Worth Surfing'])
    print("\nClassification Report:\n")
    print(report)

def save_model(model, file_path):
    joblib.dump(model, file_path)
    print(f"\nModel saved as '{file_path}'")

def plot_feature_importance(model, features, output_path):
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nFeature importance plot saved as '{output_path}'")