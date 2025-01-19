import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and preprocess the data
def load_data(file_paths, column):
    logger.info("Loading data from files...")
    combined_data = pd.DataFrame()
    for path in file_paths:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            sheet_data = xls.parse(sheet_name)
            if column in sheet_data.columns:
                combined_data = pd.concat([combined_data, sheet_data], ignore_index=True)
    logger.info("Data loading complete.")
    return combined_data

file_paths = ['1.xlsx', '2.xlsx', '3.xlsx']
data_column = 'AMOUNT'  # Replace with the column name relevant to cash flow data

logger.info("Starting data preprocessing...")
raw_data_df = load_data(file_paths, data_column)

if data_column not in raw_data_df:
    logger.warning("Specified column not found in the data. Generating random data for demonstration purposes.")
    raw_data_df = pd.DataFrame({data_column: np.random.rand(100) * 1e6})

raw_data = raw_data_df[data_column].dropna().values

# Normalize the data
logger.info("Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(raw_data.reshape(-1, 1))

# Prepare the data for LSTM
logger.info("Preparing data for LSTM...")
sequence_length = 12  # Use 12 months for prediction
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length, 0])
    y.append(data_scaled[i + sequence_length, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the LSTM model
logger.info("Building the LSTM model...")
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
logger.info("Model compilation complete.")

# Train the model
logger.info("Starting model training...")
epochs = 50
history = model.fit(X, y, epochs=epochs, batch_size=16, validation_split=0.2, verbose=1)
logger.info(f"Model training complete after {epochs} epochs.")

# Predict future cash flows
logger.info("Predicting future cash flows...")
future_steps = 12
future_predictions = []
input_seq = data_scaled[-sequence_length:]

for step in range(future_steps):
    pred = model.predict(input_seq.reshape((1, sequence_length, 1)), verbose=0)
    logger.info(f"Prediction for step {step + 1}: {pred[0, 0]}")
    future_predictions.append(pred[0, 0])
    input_seq = np.append(input_seq[1:], pred, axis=0)

# Inverse transform the predictions
logger.info("Inverse transforming predictions...")
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Combine predictions with original data
logger.info("Creating output Excel file with historical and predicted data...")
output_df = raw_data_df.copy()
output_df['Predicted Cash Flow'] = np.nan
predicted_index = range(len(output_df), len(output_df) + future_steps)
for i, index in enumerate(predicted_index):
    output_df.loc[index] = [np.nan] * (len(output_df.columns) - 1) + [future_predictions[i]]

output_file = 'cashflow_predictions.xlsx'
output_df.to_excel(output_file, index=False)
logger.info(f"Output Excel file saved at {output_file}.")

# Plot the results
logger.info("Plotting the results...")
plt.figure(figsize=(10, 6))
plt.plot(range(len(raw_data)), raw_data, label='Historical Data')
plt.plot(range(len(raw_data), len(raw_data) + future_steps), future_predictions, label='Predicted Future Data')
plt.legend()
plt.title('Cash Flow Forecasting')
plt.xlabel('Time (Months)')
plt.ylabel('Cash Flow')
plt.show()

# Save model for future use
logger.info("Saving the model...")
model.save('cashflow_lstm_model.h5')
logger.info("Model saved successfully.")
