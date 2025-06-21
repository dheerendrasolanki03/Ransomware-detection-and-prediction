import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # type: ignore

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

df = pd.read_csv('data_file.csv')

df = df.drop(['FileName', 'md5Hash'], axis=1)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df.drop('Benign', axis=1)
y = df['Benign']

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

sequence_length = 10
X_seq, y_seq = [], []
for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y[i+sequence_length])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

input_shape = X_train_seq.shape[1:] 
input_ts = Input(shape=input_shape)
x_ts = LSTM(64)(input_ts)
output_ts = Dense(1, activation='sigmoid')(x_ts)

lstm_model = Model(inputs=input_ts, outputs=output_ts)
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training LSTM for future prediction...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

loss, accuracy = lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
import os
os.makedirs("models", exist_ok=True)

lstm_model.save('models/future_lstm.h5')

print("LSTM and Autoencoder models saved successfully.")
