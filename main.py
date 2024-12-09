import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('archive/BTC-Daily.csv')

# Convert date to datetime format and set it as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Select relevant features and target variable
features = data[['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']]
target = data['close']

# Feature Selection: Correlation Matrix
plt.figure(figsize=(10, 6))
correlation_matrix = features.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM (3D shape)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Define LSTM Model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define Dense Model
def create_dense_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create models
lstm_model = create_lstm_model()
dense_model = create_dense_model()

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train LSTM Model
lstm_history = lstm_model.fit(X_train_lstm, y_train.values,
                               epochs=100,
                               batch_size=32,
                               validation_data=(X_test_lstm, y_test.values),
                               callbacks=[early_stopping],
                               verbose=1)

# Train Dense Model
dense_history = dense_model.fit(X_train_scaled, y_train.values,
                                 epochs=100,
                                 batch_size=32,
                                 validation_data=(X_test_scaled, y_test.values),
                                 callbacks=[early_stopping],
                                 verbose=1)

# Predictions
lstm_predictions = lstm_model.predict(X_test_lstm)
dense_predictions = dense_model.predict(X_test_scaled)

# Calculate MSE for both models
lstm_mse = mean_squared_error(y_test.values, lstm_predictions)
dense_mse = mean_squared_error(y_test.values, dense_predictions)

print(f"LSTM Model MSE: {lstm_mse}")
print(f"Dense Model MSE: {dense_mse}")

# Visualization of predictions vs actual prices
plt.figure(figsize=(14,7))
plt.plot(y_test.index, y_test.values, label='Actual Prices', color='blue')
plt.plot(y_test.index, lstm_predictions.flatten(), label='LSTM Predictions', color='orange')
plt.plot(y_test.index, dense_predictions.flatten(), label='Dense Predictions', color='green')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()