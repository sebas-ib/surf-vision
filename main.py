import sys
import os
import argparse
from data_preprocessing import load_data, preprocess_data, select_features
from data_utils import encode_target, split_data
from model_training import train_model
from model_utils import evaluate_model, plot_feature_importance, save_model
from config import spot_files, models  # Import the dictionaries

def main():
    # Use argparse for command-line arguments
    parser = argparse.ArgumentParser(
        description='Train and evaluate surf prediction models for different surf spots.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=models.keys(),  # Use keys from the models dictionary
        help=(
            'Model to use for prediction (default: random_forest). '
            'Available models: ' + ', '.join(models.keys()) + '.'
        )
    )
    parser.add_argument(
        '--spot',
        type=str,
        required=True,
        choices=spot_files.keys(),  # Use keys from the spot_files dictionary
        help=(
            'Surf spot to use for data. '
            'Available spots: ' + ', '.join(spot_files.keys()) + '.'
        )
    )
    args = parser.parse_args()

    model_type = args.model
    spot = args.spot

    # Get data file for the selected spot
    data_file = spot_files[spot]

    # Load and preprocess data
    data = load_data(data_file)
    data = preprocess_data(data)
    X, y = select_features(data)
    features = X.columns # save features for plotting

    # Encode target & divide data into test & train sets
    y = encode_target(y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Get the model instance
    model = models[model_type]

    # Train and evaluate model
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Plot feature importance if available
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, features, f'results/{model_type}/feature_importance_{spot}.png')

    # Ensure the directory exists
    os.makedirs(f'results/{model_type}', exist_ok=True)

    # Save the model
    save_model(model, f'results/{model_type}/surf_prediction_model_{spot}.joblib')

if __name__ == "__main__":
    main()