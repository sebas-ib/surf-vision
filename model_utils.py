import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, output_dir):
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

    # Get classification report as a dictionary
    report_dict = classification_report(y_test, y_pred, target_names=['Not Worth Surfing', 'Worth Surfing'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    # Round metrics to two decimals
    report_df[['precision', 'recall', 'f1-score', 'support']] = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)

    # Create a DataFrame for accuracy
    accuracy_df = pd.DataFrame({'accuracy': [accuracy]}).round(2)

    # Save the report and accuracy to CSV files
    report_df.to_csv(f'{output_dir}/classification_report.csv')
    accuracy_df.to_csv(f'{output_dir}/accuracy.csv')

    print("\nClassification Report:\n")
    print(report_df)

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