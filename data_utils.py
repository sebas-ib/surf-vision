import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_non_numeric_features(data):
    '''
    Encode non-numeric features using sklearn's LabelEncoder
    '''
    le = LabelEncoder()

    for column in data.select_dtypes(include=['object']).columns:
        print(f"Encoding {column}...")
        data[column] = le.fit_transform(data[column])

    print("Features encoded\n")

    return data

def split_data(X, y, test_size=0.2, random_state=42):
    '''
    Split data into training and test sets using sklearn's train_test_split
    '''
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    '''
    Scale features using sklearn's StandardScaler
    '''
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled")

    return X_train_scaled, X_test_scaled