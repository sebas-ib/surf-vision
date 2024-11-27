import pandas as pd

def load_data(path):
    data = pd.read_csv(path)

    print("Data loaded")

    return data

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month

    print("Data preprocessed")

    return data

def select_features(data):
    features = ['Time', 
                'Wave Height (ft)', 
                'Primary Swell (ft)', 
                'Primary Swell (seconds)', 
                'Wind (kts)', 
                'Wave Energy', 
                'Consistency', 
                'Month'
    ]
    
    X = data[features]
    y = data['w/nw']

    print("Features selected")

    return X, y