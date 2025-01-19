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
def load_data(file_paths, columns):
    logger.info("Loading data from files...")
    combined_data = pd.DataFrame()
    for path in file_paths:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            sheet_data = xls.parse(sheet_name)
            sheet_data.columns = sheet_data.columns.str.strip().str.lower()  # Normalize column names
            for column in columns:
                if column.lower() in sheet_data.columns:
                    if combined_data.empty:
                        combined_data = sheet_data
                    else:
                        combined_data = pd.concat([combined_data, sheet_data], ignore_index=True)
    logger.info("Data loading complete.")
    return combined_data

file_paths = ['1.xlsx', '2.xlsx', '3.xlsx']
required_columns = ['description', 'qty', 'rate', 'previous amount', 'amount']  # Include all relevant columns

logger.info("Starting data preprocessing...")
raw_data_df = load_data(file_paths, required_columns)

if raw_data_df.empty:
    logger.error("No data found in the files. Exiting...")
    exit()

# Normalize the data
logger.info("Normalizing data...")
scalers = {}
data_scaled = {}
for column in required_columns:
    if column.lower() in raw_data_df.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled[column] = scaler.fit_transform(raw_data_df[column.lower()].fillna(0).values.reshape(-1, 1))
        scalers[column] = scaler

# Combine all scaled columns for input
logger.info("Combining scaled columns for LSTM input...")
data_combined = np.column_stack([data_scaled[col] for col in required_columns if col in data_scaled])

# Prepare the data for LSTM
sequence_length = 12
X, y = [], []
for i in range(len(data_combined) - sequence_length):
    X.append(data_combined[i:i + sequence_length, :])
    y.append(data_combined[i + sequence_length, 0])  # Predicting based on 'amount'
X, y = np.array(X), np.array(y)

# Build the LSTM model
logger.info("Building the LSTM model...")
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[2])),
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
input_seq = data_combined[-sequence_length:]

for step in range(future_steps):
    pred = model.predict(input_seq.reshape((1, sequence_length, X.shape[2])), verbose=0)
    logger.info(f"Prediction for step {step + 1}: {pred[0, 0]}")
    future_predictions.append(pred[0, 0])
    new_row = np.append(input_seq[1:], pred, axis=0)
    input_seq = np.append(input_seq[1:], new_row[-1:], axis=0)

# Inverse transform the predictions
logger.info("Inverse transforming predictions...")
inverse_predictions = scalers['amount'].inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Combine predictions with original data
logger.info("Creating output Excel file with historical and predicted data...")
output_df = raw_data_df.copy()
output_df['Predicted Cash Flow'] = np.nan
predicted_index = range(len(output_df), len(output_df) + future_steps)
for i, index in enumerate(predicted_index):
    row = {col: np.nan for col in raw_data_df.columns}
    row['Predicted Cash Flow'] = inverse_predictions[i]
    output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)

output_file = 'cashflow_predictions.xlsx'
output_df.to_excel(output_file, index=False)
logger.info(f"Output Excel file saved at {output_file}.")

# Plot the results
logger.info("Plotting the results...")
plt.figure(figsize=(10, 6))
plt.plot(range(len(raw_data_df)), raw_data_df['amount'].fillna(0), label='Historical Data')
plt.plot(range(len(raw_data_df), len(raw_data_df) + future_steps), inverse_predictions, label='Predicted Future Data')
plt.legend()
plt.title('Cash Flow Forecasting')
plt.xlabel('Time (Months)')
plt.ylabel('Cash Flow')
plt.show()

# Save model for future use
logger.info("Saving the model...")
model.save('cashflow_lstm_model.h5')
logger.info("Model saved successfully.")
