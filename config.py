from logistic_regression import LogisticRegression
from random_forest import RandomForestClassifier

# Map spot names to data file paths
spot_files = {
    'blacks_beach': 'swell-data/Blacks_Beach_North_San_Diego_County_swell_data.csv',
    'cardiff_reef': 'swell-data/Cardiff_Reef_North_San_Diego_County_swell_data.csv',
    'malibu_north': 'swell-data/Malibu_North_Los_Angeles_County_swell_data.csv',
    'oceanside_harbour': 'swell-data/Oceanside_Harbor_North_San_Diego_County_swell_data.csv',
    'oceanside_pier': 'swell-data/Oceanside_Pier_North_San_Diego_County_swell_data.csv',
    'san_onofre': 'swell-data/San_Onofre_South_Orange_County_swell_data.csv'
    # Add more spots here if needed
}

# Available models
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'logistic_regression': LogisticRegression(alpha=0.01, num_iters=1000),
    # Add more models here if needed
}