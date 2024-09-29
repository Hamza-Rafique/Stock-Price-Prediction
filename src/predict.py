import numpy as np
import joblib
from keras.models import load_model

def predict_with_linear_regression(X):
    """Predict using the trained Linear Regression model."""
    model = joblib.load('models/linear_regression.pkl')
    predictions = model.predict(X)
    return predictions

def predict_with_decision_tree(X):
    """Predict using the trained Decision Tree model."""
    model = joblib.load('models/decision_tree.pkl')
    predictions = model.predict(X)
    return predictions

def predict_with_lstm(X):
    """Predict using the trained LSTM model."""
    model = load_model('models/lstm_model.h5')
    X_lstm = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    predictions = model.predict(X_lstm)
    return predictions

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data

    # Load the test data
    features, target, scaler = load_and_preprocess_data('data/stock_prices.csv')
    
    # Predict with each model
    print("Linear Regression Predictions:", predict_with_linear_regression(features))
    print("Decision Tree Predictions:", predict_with_decision_tree(features))
    print("LSTM Predictions:", predict_with_lstm(features))
