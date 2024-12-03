def train_model(model, X_train, y_train):
    '''
    Train the provided model using the training data
    '''
    print(f"Training {model.__class__.__name__}...")

    model.fit(X_train, y_train)

    print("Model trained")
    
    return model