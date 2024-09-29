from sklearn.metrics import mean_squared_error

def evaluate_model(predictions, true_values):
    """Evaluate model performance using Mean Squared Error."""
    mse = mean_squared_error(true_values, predictions)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    from predict import predict_with_linear_regression, predict_with_decision_tree, predict_with_lstm
    from data_preprocessing import load_and_preprocess_data
    
    # Load the test data
    features, target, scaler = load_and_preprocess_data('data/stock_prices.csv')
    
    # Evaluate Linear Regression
    predictions_lr = predict_with_linear_regression(features)
    evaluate_model(predictions_lr, target)
    
    # Evaluate Decision Tree
    predictions_dt = predict_with_decision_tree(features)
    evaluate_model(predictions_dt, target)
    
    # Evaluate LSTM
    predictions_lstm = predict_with_lstm(features)
    evaluate_model(predictions_lstm, target)
