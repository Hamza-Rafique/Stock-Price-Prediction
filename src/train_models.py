import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib
import numpy as np
def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/linear_regression.pkl')
    return model

def train_decision_tree(X_train, y_train):
    """Train a Decision Tree Regressor."""
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/decision_tree.pkl')
    return model

def train_lstm(X_train, y_train):
    """Train an LSTM model."""
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)
    model.save('models/lstm_model.h5')
    return model

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data

    # Load the data
    features, target, scaler = load_and_preprocess_data('data/stock_prices.csv')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the models
    train_linear_regression(X_train, y_train)
    train_decision_tree(X_train, y_train)
    train_lstm(X_train, y_train)
