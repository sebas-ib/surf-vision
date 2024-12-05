import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_single_spot_data(path):
    '''
    Load the csv data from the specified path.
    '''
    data = pd.read_csv(path)

    print("Data loaded\n")

    return data

def load_all_spots_data(file_paths):
    '''
    Load and concatenate CSV data from multiple specified paths,
    ensuring headers are handled correctly.
    '''
    data_list = []
    for path in file_paths.values():
        data = pd.read_csv(path)
        data_list.append(data)

    data = pd.concat(data_list, ignore_index=True)

    print("Data loaded and concatenated\n")

    return data

def preprocess_data(data):
    '''
    Preprocess the data. More preprocessing steps can be added here.
    '''
    # Trim all string columns to erase leading/trailing spaces
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Get Month from 'Date' column
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month

    print("Data preprocessed\n")

    return data

def select_features(data, include_surf_break=False):
    '''
    Select the features to use for training the model.
    '''
    features = [
        'Time', 
        'Wave Height (ft)', 
        'Primary Swell (ft)', 
        'Primary Swell (seconds)', 
        'Wind (kts)', 
        'Wave Energy', 
        'Consistency', 
        'Month'
    ]

    if include_surf_break:
        features.append('Surf Break')

    X = data[features]
    y = data['w/nw']

    print("Selected features:", features, "\n")

    return X, y