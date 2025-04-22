
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def forecast_data(row, forecast_out):
    # Preprocess the data
    X = np.array(row).reshape(-1, 1)
    scaler = preprocessing.StandardScaler()  # Use StandardScaler to standardize
    X_scaled = scaler.fit_transform(X)       # Scale the data
    
    X_forecast = X_scaled[-forecast_out:]
    X_scaled = X_scaled[:-forecast_out]
    y = np.array(row.shift(-forecast_out))
    y = y[:-forecast_out]
    
    # Reshape for LSTM: (samples, timesteps, features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, 1))
    X_forecast = X_forecast.reshape((X_forecast.shape[0], 1, 1))
    
    # Train-test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.2)
    
    # Define the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, 1)),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    # Evaluate the model (confidence can be calculated as R-squared manually)
    confidence = model.evaluate(X_test, y_test, verbose=0)
    
    # Forecast future data
    forecast_prediction_scaled = model.predict(X_forecast).flatten()
    
    # Inverse transform the forecasted data to original scale
    forecast_prediction = scaler.inverse_transform(forecast_prediction_scaled.reshape(-1, 1)).flatten()
    
    return forecast_prediction.tolist()

