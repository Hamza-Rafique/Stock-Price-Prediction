import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    """Load and preprocess the stock price data."""
    data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    
    # Features: Open, High, Low, Volume
    features = data[["Open", "High", "Low", "Volume"]]
    target = data["Close"]
    
    # Scale the data to the range (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target.values, scaler

if __name__ == "__main__":
    # Load the dummy data
    features, target, scaler = load_and_preprocess_data('data/stock_prices.csv')
    print(features[:5])  # Check the processed data
