def train_model(model, X_train, y_train):
    print(f"Training {model.__class__.__name__}...")

    model.fit(X_train, y_train)

    print("Model trained")
    
    return model